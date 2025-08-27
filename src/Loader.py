import os
import json
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import torch
from .answerEncoder import answerNumber

class Loader(Dataset):
    def __init__(
            self,
            config,
            DataConfig,
            seqEncoder,
            img_size,
            textHead,
            imageHead,
            train=True,
            transform=None,
    ):#定义类的初始化方法，接受配置参数，数据配置，编码器，图像大小，文本，图像头，训练模式标志，图形是否变换等信息
        self.config = config
        self.img_size = img_size
        self.Encoder = seqEncoder
        self.textHead = textHead
        self.imgHead = imageHead
        self.imgFolder = config["new_data_path"]
        self.questions_file = DataConfig["questionsJSON"]
        self.images_file = DataConfig["imagesJSON"]
        self.imageFile = config["DataConfig"]["images_path"]
        self.imgSource = config["DataConfig"]["sourceMask_path"]
        self.imgTarget = config["DataConfig"]["targetMask_path"]
        self.imgSeg = config["DataConfig"]["seg_path"]
        self.imgBackground = config["DataConfig"]["backgroundMask_path"]
        self.train = train
        self.transform = transform
        self.addMask = config["add_mask"]
        self.answerEncoder = answerNumber(config, config["DataConfig"]["answersJson"])

        with open(self.questions_file) as json_data:
            self.questionsJSON = json.load(json_data)
        with open(self.images_file) as json_data:
            self.imagesJSON = json.load(json_data)

        self.imageActive = [img["id"] for img in self.imagesJSON["images"] if img["active"]]
        self.questionActive = [q["id"] for q in self.questionsJSON["questions"] if q["active"]]
        self.length = len(self.questionActive)
        self.questions = self.questionsJSON["questions"]
#取得活跃的问题和图像，并且得到活跃问题的数量，建立只存在活跃问题的变量，活跃图像，整个question（包含多个键值对）存在questions中
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        question = self.questions[idx]#这个question包括在question数据集中对应id编号那个列表的所有属性
#获取一个特定的问题
        image = question['img_id']
        img = Image.open(os.path.join(self.imageFile, str(image) + ".png"))
#打开图像
        if img.mode == "RGBA":
            img = img.convert("RGB")#将图像转化为RGB模式
        img = np.array(img)#将图像转化为数组
        target = np.array(Image.open(os.path.join(self.imgTarget, str(image) + ".png")))
        source = np.array(Image.open(os.path.join(self.imgSource, str(image) + ".png")))
        #打开存在于这几个位置的图像文件，并全部转化为numpy数组
        background = np.array(Image.open(os.path.join(self.imgBackground, str(image) + ".png")))
        seg = np.array(Image.open(os.path.join(self.imgSeg, str(image) + ".png")))
#
        source = source[:, :, np.newaxis]
        target = target[:, :, np.newaxis]
        background = background[:, :, np.newaxis]
        seg = seg[:, :, np.newaxis]
#把这些图像全部增加一个维度，使其能够处理具有单一颜色通道的图像，本来都是二维
        source_mask = source + background * 0.1
        target_mask = target + background * 0.1#对这两个图像进行更改，数据包含元图像和background图像的数据
        if self.addMask:
            background_mask = background * 0.1 + source * 0.7 + target * 0.9
        else:
            background_mask = background * 0.1
#确定background_mask的值
        # mask = np.concatenate((source_mask, target_mask, background_mask), axis=-1).astype(np.uint8)
        # The mask has not been normalized.
        # This may be modified in future versions, but currently this method works better than directly normalizing the mask
        mask = np.concatenate((source_mask, target_mask, background_mask), axis=-1)
        sourceImage = T.ToTensor()(img)#这个变量，将通常是 H x W x C 的形状，其中 H 是高度，W 是宽度，C 是颜色通道数）
        # 转换为 C x H x W 形状的浮点张量。同时，它还将像素值从 [0, 255] 范围映射到 [0.0, 1.0] 范围
        mask = self.transform["mask"](mask).float()
        imgT = self.transform["image"](img.copy())
#对掩码和原始图像的副本进行特定的处理
        Question = self.Encoder.encode(question['question'], question=True)
        #对问题进行编码
        if self.textHead == "siglip_512":
            Question["input_ids"] = (#在这个分词器中，input_id是自动生成的
                torch.as_tensor(np.array(Question["input_ids"])).long().squeeze(0)#将包含tokenid的列表转化成一维的长整形张量
            )
        elif self.textHead in ['skipthoughts', '2lstm','lstm']:
            tempQ = torch.as_tensor(np.array(Question)).long().squeeze(0)
            Question = tempQ
        else:
            Question["input_ids"] = (
                torch.as_tensor(np.array(Question["input_ids"])).long().squeeze(0)
            )
            Question["attention_mask"] = (
                torch.as_tensor(np.array(Question["attention_mask"])).long().squeeze(0)
            )
#全转成一维的长整型张量
        answer = self.answerEncoder.encode(question['type'], question['answer'])
        answer = torch.as_tensor(np.array(answer)).long()#获取答案对应的答案编码
        #answer = self.answerEncoder.encode("1", "Yes")返回1
        if self.train:
            return (Question, answer, imgT, mask)
        else:
            return (Question, answer, imgT, question["type"], mask, sourceImage)
