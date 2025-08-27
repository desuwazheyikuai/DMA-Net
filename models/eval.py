import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 修正后的推理代码
def detect_tampering(image_path, config, question_vocab, device="cuda"):
    # 1. 根据训练配置构建模型
    model = CDModel(
        config,
        question_vocab.getVocab(),
        input_size=config["image_resize"],
        textHead=config["textHead"],
        imageHead=config["imageHead"],
        trainText=config["trainText"],
        trainImg=config["trainImg"],
    )
    model.load_state_dict(torch.load('bestValLoss.pth'))  # 加载训练好的权重
    model.to(device).eval()

    # 2. 图像预处理 (保持与训练一致)
    def preprocess_image(image_path):
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((config["image_resize"], config["image_resize"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return transform(img)

    # 3. 创建伪输入 (用于缺失的数据)
    image_tensor = preprocess_image(image_path).unsqueeze(0).to(device)
    h, w = image_tensor.shape[2:]

    # 创建空白掩码 (尺寸与图像一致)
    dummy_mask = torch.zeros((1, 1, h, w)).to(device)

    # 创建问题输入 (需要根据实际question_vocab实现)
    dummy_question = torch.zeros((1, config["max_question_len"])).long().to(device)

    # 4. 运行推理
    with torch.no_grad():
        pred, pred_mask = model(
            image_tensor,
            dummy_question,
            dummy_mask
        )

    # 5. 后处理 (与训练中的归一化一致)
    pred_mask = pred_mask.squeeze().cpu().numpy()

    if not config.get('normalize', False):
        pred_mask = pred_mask / 255.0  # 与训练中的处理一致

    # 6. 调整掩码尺寸以匹配原始图像
    original_img = np.array(Image.open(image_path).convert('RGB'))
    resized_mask = np.array(Image.fromarray(pred_mask).resize(
        (original_img.shape[1], original_img.shape[0]),
        Image.BILINEAR
    ))

    # 7. 可视化 (确保图像和掩码尺寸匹配)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(resized_mask, cmap='viridis')
    plt.title('Predicted Tamper Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('tamper_result.jpg', bbox_inches='tight')

    print(f"Results saved to tamper_result.jpg")

    return pred_mask, resized_mask


# 使用示例
if __name__ == "__main__":
    # 需要从训练中获取的配置和词汇表
    from config import get_config  # 假设的配置模块
    from vocabulary import QuestionVocabulary  # 假设的词汇表模块

    config = get_config()
    question_vocab = QuestionVocabulary()  # 需要实际初始化
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detect_tampering("test_image.jpg", config, question_vocab, device)