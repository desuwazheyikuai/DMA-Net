import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from src import Logger, train_mask_model
from transformers import optimization
from tqdm import tqdm
import time
import wandb
from test import test_model
import torch.nn.functional as F
from models import maskModel, CDModel


def train(_config, train_dataset, val_dataset, test_dataset, device, question_vocab):
    use_wandb = _config['use_wandb']
    wandb_step, wandb_epoch = None, None
    if use_wandb:
        wandb.login(key=_config["wandbKey"])
        wandb_epoch = None
        wandb_step = wandb.init(
            config=_config,
            project=_config["project"] + "_steps",
            name=_config["wandbName"],
            job_type=_config["job_type"],
            reinit=True,
        )
    start = time.time()

    textHead = _config["textHead"]
    imageHead = _config["imageHead"]
    trainText = _config["trainText"]
    trainImg = _config["trainImg"]
    image_size = _config["image_resize"]
    batch_size = _config["batch_size"]
    oneStep = _config["one_step"]
    opts = _config["opts"]
    num_epochs = _config["num_epochs"]
    learning_rate = _config["learning_rate"]
    saveDir = _config["saveDir"]
    miniStep = _config["steps"]
    is_scheduler = _config["scheduler"]
    num_workers = _config["num_workers"]
    pin_memory = _config["pin_memory"]
    persistent_workers = _config["persistent_workers"]
    classes = _config["question_classes"]

    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)
    log_file_name = (
            saveDir + "log-" + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + ".log"
    )#日志文件的记录和保存
    logger = Logger(log_file_name)
    logger.info(f"saveDir: {saveDir}")
    bestVal = 9999999
    bestAcc = 0#损失和验证准确率
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(batch_size, 2),
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    optimizer_mask = None
    if oneStep:
        model = CDModel(
            _config,
            question_vocab.getVocab(),
            input_size=image_size,
            textHead=textHead,
            imageHead=imageHead,
            trainText=trainText,
            trainImg=trainImg,
        )
        if opts:#opts变量用于指示是否需要为maskNet部分的参数单独创建优化器
            maskNet_params = [p for p in model.maskNet.parameters() if p.requires_grad]
            #包含所有需要更新的参数
            maskNet_ids = {id(p) for p in model.maskNet.parameters()}
            #所有掩码的表示符
            other_params = [
                p
                for p in model.parameters()#返回模型中所有需要梯度更新的参数（权重，偏置）
                if id(p) not in maskNet_ids
                if p.requires_grad
            ]#在P满足标准之后，将符合条件的P写入到otherparams(生成器)
            optimizer_mask = torch.optim.Adam(maskNet_params, lr=1e-3)

            optimizer = torch.optim.Adam(
                other_params,
                lr=learning_rate,
                weight_decay=_config["weight_decay"],
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                weight_decay=_config["weight_decay"],
            )
            if _config["opt"] == "SGD":
                optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=learning_rate,
                    weight_decay=_config["weight_decay"],
                    momentum=0.9,
                )
    else:
        mask_model = maskModel(_config).to(device)
        train_mask_model(
            _config,
            mask_model,
            train_loader,
            len(train_dataset),
            val_loader,
            len(val_dataset),
            device,
            logger,
        )
        model = CDModel(
            _config,
            question_vocab.getVocab(),
            input_size=image_size,
            textHead=textHead,
            imageHead=imageHead,
            trainText=trainText,
            trainImg=trainImg,
        )
        for param in model.maskNet.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            #filter 函数用于筛选模型中需要梯度更新的参数
            lr=learning_rate,
            weight_decay=_config["weight_decay"],
        )

    scheduler = None
    mask_scheduler = None#（scheduler学习率调度器）
    if is_scheduler:
        lr_end = _config["end_learning_rate"]
        if _config["CosineAnnealingLR"]:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=30, eta_min=lr_end
            )#如果配置文件中 CosineAnnealingLR 为 True，则使用 CosineAnnealingLR 调度器。
            # 这个调度器会按照余弦函数的形状调整学习率，从初始学习率逐渐降低到 eta_min（lr_end）。
            # T_max 是学习率从最高值降到最低值所需的最大 epoch 数
        elif _config["warmUp"]:
            scheduler = optimization.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=5,
                num_training_steps=30,
                lr_end=lr_end,
                power=2,#会在训练的前 num_warmup_steps 个 epoch 中逐渐增加学习率，
                # 然后按照多项式衰减到 lr_end
            )
        if oneStep and opts:
            mask_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer_mask, T_max=30, eta_min=1e-4#学习率最终下降到这个
            )

    criterion = torch.nn.CrossEntropyLoss()
    (
        trainLoss,#存储训练过程中的总损失
        trainAccLoss,#存储训练过程中的准确率损失
        trainMaeLoss,#存储训练过程中的平均绝对误差（MAE）损失
        trainRmseLoss,#存储训练过程中的均方根误差（RMSE）损失

        valLoss,#存储验证过程中的总损失
        valAccLoss,#
        valMaeLoss,#
        valRmseLoss,#

        acc,#存储准确率
    ) = ([], [], [], [], [], [], [], [], [])

    accPerQuestionType = {str(i): [] for i in range(1, classes + 1)}
    logger.info(
        f"Started training... total epoch: {num_epochs}, batch-size: {int(batch_size * miniStep)}, step: {miniStep}"
    )#在这个batch-size是经过批次导致更新参数，因为这个ministep被设置成1，则batch-size就是32
    called = False
    model.to(device)
    steps = 0
    for epoch in range(num_epochs):
        # train
        model.train()
        accLoss, maeLoss, rmseLoss = 0, 0, 0
        #expert_count=np.zeros(num_of_experts),就是六位数字
        logger.info(f"Epoch {epoch}")
        t1 = time.time()
        if oneStep and epoch >= _config["thread_epoch"] and not called:
            called = True
            for param in model.maskNet.parameters():
                param.requires_grad = False
        for i, data in tqdm(
                enumerate(train_loader, 0),
                total=len(train_loader),
                ncols=100,
                mininterval=1,
        ):
            question, answer, image, mask = data
            pred, pred_mask= model(
                image.to(device), question.to(device), mask.to(device), epoch=epoch#轮数，看是否冻结掩码参数
            )
            answer = answer.to(device)
            mae = F.l1_loss(mask.to(device), pred_mask)
            mse = F.mse_loss(mask.to(device), pred_mask)
            rmse = torch.sqrt(mse)
            acc_loss = criterion(pred, answer)

            loss = 0.3 * rmse + 0.7 * acc_loss
            # The ground truth of mask has not been normalized. (Which is intuitively weird)
            # This may be modified in future versions, but currently this method works better than directly normalizing the mask
            if not _config['normalize']:
                mae = mae / 255
                rmse = rmse / 255
            step_acc = acc_loss.cpu().item()
            step_mae = mae.cpu().item()
            step_rmse = rmse.cpu().item()

            if epoch == 0 and use_wandb:
                wandb_step.log(
                    {
                        "step loss": step_rmse + step_mae + step_acc,
                        "step acc loss": step_acc,
                        "step mae loss": step_mae,
                        "step rmse loss": step_rmse,

                    },
                    step=steps,#如果一个epoch包含100个batch，那么在完成每个batch的训练后，步数会增加1，从1到100
                )
            steps += 1
            accLoss += step_acc * image.shape[0]
            maeLoss += step_mae * image.shape[0]
            rmseLoss += step_rmse * image.shape[0]
            #步的损失乘当前batch的样本数量,！！！计算总损失
            # --------------------- L1 ---------------------------
            if _config["L1Reg"]:
                L1_reg = 0
                for param in model.parameters():
                    L1_reg += torch.sum(torch.abs(param))
                loss = (loss + L1_reg * 1e-7) / miniStep
            # -----------------------------------------------------
            else:
                loss = loss / miniStep
            loss.backward()
            if (i + 1) % miniStep == 0:
                if oneStep and opts and epoch < _config["thread_epoch"]:
                    optimizer_mask.step()
                    optimizer_mask.zero_grad()
                optimizer.step()
                optimizer.zero_grad()
#在循环的最后，accLoss等已经累计了全部的损失
        trainAccLoss.append(accLoss / len(train_dataset))
        trainMaeLoss.append(maeLoss / len(train_dataset))
        trainRmseLoss.append(rmseLoss / len(train_dataset))

        trainLoss.append(
            trainAccLoss[epoch] + trainRmseLoss[epoch] + trainMaeLoss[epoch]
        )
        t2 = time.time()
        lr = optimizer.param_groups[0]["lr"]#构造优化器时提供了参数组。每个参数组可以包含模型中的一部分参数，并且设置学习率
        logger.info(
            f"Training: epoch {epoch}, train loss: {trainLoss[epoch]:.5f}, acc loss : {trainAccLoss[epoch]:.5f}, "
            f"mae loss: {trainMaeLoss[epoch]:.5f}, rmse loss: {trainRmseLoss[epoch]:.5f},lr: {lr}\n"

        )
        if use_wandb:
            wandb_step.finish()

        if epoch == 0 and use_wandb:
            wandb_epoch = wandb.init(
                config=_config,
                project=_config["project"],
                name=_config["wandbName"],
                job_type=_config["job_type"],
                reinit=True,
            )
        if use_wandb:
            wandb_epoch.log(
                {
                    "train loss": trainLoss[epoch],
                    "train acc loss": trainAccLoss[epoch],
                    "train mae loss": trainMaeLoss[epoch],
                    "train rmse loss": trainRmseLoss[epoch],

                    "learning rate": lr,
                    "train time cost": t2 - t1,
                },
                step=epoch,
            )
        if is_scheduler:
            scheduler.step()
            if oneStep and opts and epoch < _config["thread_epoch"]:
                mask_scheduler.step()
        # --------------------validation-------------------
        v1 = time.time()
        logger.info(f"Validation:")
        with torch.no_grad():
            model.eval()
            accLoss, maeLoss, rmseLoss = 0, 0, 0

            countQuestionType = {str(i): 0 for i in range(1, classes + 1)}
            rightAnswerByQuestionType = {str(i): 0 for i in range(1, classes + 1)}

            for i, data in tqdm(
                    enumerate(val_loader, 0),
                    total=len(val_loader),
                    ncols=100,
                    mininterval=1,
            ):
                question, answer, image, type_str, mask, image_original = data
                pred, pred_mask = model(
                    image.to(device), question.to(device), mask.to(device)
                )
                answer = answer.to(device)
                mae = F.l1_loss(mask.to(device), pred_mask)
                mse = F.mse_loss(mask.to(device), pred_mask)
                rmse = torch.sqrt(mse)


                # The ground truth of mask has not been normalized. (Which is intuitively weird)
                # This may be modified in future versions, but currently this method works better than directly normalizing the mask
                if not _config['normalize']:
                    mae = mae / 255
                    rmse = rmse / 255
                #当前批次的总共损失
                acc_loss = criterion(pred, answer)
                accLoss += acc_loss.cpu().item() * image.shape[0]
                maeLoss += mae.cpu().item() * image.shape[0]
                rmseLoss += rmse.cpu().item() * image.shape[0]
                answer = answer.cpu().numpy()
                pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
#沿着行找到最大值的索引，就是预测类别
                for j in range(answer.shape[0]):
                    countQuestionType[type_str[j]] += 1
                    if answer[j] == pred[j]:
                        rightAnswerByQuestionType[type_str[j]] += 1
#遍历每个样本，统计每种问题类型出现的次数。
#同时，统计每种问题类型中预测正确的次数，以便后续计算每种问题类型的准确率
            valAccLoss.append(accLoss / len(val_dataset))
            valMaeLoss.append(maeLoss / len(val_dataset))
            valRmseLoss.append(rmseLoss / len(val_dataset))
            valLoss.append(valAccLoss[epoch] + valRmseLoss[epoch] + valMaeLoss[epoch])
            if valLoss[epoch] < bestVal:
                bestVal = valLoss[epoch]
                # torch.save(model, f"{saveDir}bestValiLoss.pth")
                torch.save(model.state_dict(), f"{saveDir}bestValLoss.pth")
            logger.info(
                f"Epoch {epoch} , val loss: {valLoss[epoch]:.5f}, acc loss : {valAccLoss[epoch]:.5f}, "
                f"mae loss: {valMaeLoss[epoch]:.5f}, rmae loss: {valRmseLoss[epoch]:.5f}, "

            )

            numQuestions = 0
            numRightQuestions = 0
            logger.info("Acc:")
            subclassAcc = {}
            for type_str in countQuestionType.keys():
                if countQuestionType[type_str] > 0:
                    accPerQuestionType[type_str].append(
                        rightAnswerByQuestionType[type_str]
                        * 1.0
                        / countQuestionType[type_str]
                    )
                else:
                    accPerQuestionType[type_str].append(0)
                numQuestions += countQuestionType[type_str]
                numRightQuestions += rightAnswerByQuestionType[type_str]
                subclassAcc[type_str] = tuple(
                    (countQuestionType[type_str], accPerQuestionType[type_str][epoch])
                )
            logger.info(
                "\t".join(
                    [
                        f"{key}({subclassAcc[key][0]}) : {subclassAcc[key][1]:.5f}"
                        for key in subclassAcc.keys()
                    ]
                )
            )

            # ave acc
            acc.append(numRightQuestions * 1.0 / numQuestions)
            if acc[epoch] > bestAcc:
                bestAcc = acc[epoch]
                torch.save(model.state_dict(), f"{saveDir}bestValAcc.pth")
            AA = 0
            for key in subclassAcc.keys():
                if use_wandb:
                    wandb_epoch.log(
                        {"val " + key + " acc": subclassAcc[key][1]}, step=epoch
                    )
                AA += subclassAcc[key][1]
            AA = AA / len(subclassAcc)

            v2 = time.time()
            logger.info(f"overall acc: {acc[epoch]:.5f}\taverage acc: {AA:.5f}")
            if use_wandb:
                wandb_epoch.log(
                    {
                        "val overall acc": acc[epoch],
                        "val average acc": AA,
                        "val loss": valLoss[epoch],
                        "val acc loss": valAccLoss[epoch],
                        "val mae loss": valMaeLoss[epoch],
                        "val rmse loss": valRmseLoss[epoch],

                        "validation time cost": v2 - v1,
                    },
                    step=epoch,
                )
        torch.save(model.state_dict(), f"{saveDir}lastValModel.pth")
    test_model(
        _config,
        model,
        test_loader,
        len(test_dataset),
        device,
        logger,
        wandb_epoch,
        num_epochs,
    )
    end = time.time()
    logger.info(f"time used: {end - start} s")

    return model