import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import models


class RouterGate(nn.Module):
    def __init__(self, config):
        super(RouterGate, self).__init__()
        self.config = config
        self.output_dim = int(config["attnConfig"]["embed_size"] / 4)  # 局部特征维度
        self.cnnEncoder = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.cnnEncoder.fc.in_features
        self.cnnEncoder.fc = nn.Linear(num_ftrs, self.output_dim)
        self.cnnEncoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 适配单通道输入
        self.global_feature_proj = nn.Linear(config["attnConfig"]["embed_size"], self.output_dim)
        self.activation = nn.SiLU()  # 激活函数

    def forward(self, source, target, background, image,text=None):
        s = self.cnnEncoder(source)  # (batch_size, output_dim)
        t = self.cnnEncoder(target)
        b = self.cnnEncoder(background)
        img = self.global_feature_proj(image)  # (batch_size, output_dim)
        img = self.activation(img)

        visionFeatures= torch.cat([s, t, b, img], dim=1)  # (batch_size, 4 * output_dim)
        return visionFeatures