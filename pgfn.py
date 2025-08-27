import torch
import torch.nn as nn
import torch.nn.functional as F


class TextImageFusion(nn.Module):


    def __init__(self, dim=768, reduction_ratio=4, dropout=0.1):
        super(TextImageFusion, self).__init__()
        self.dim = dim
        self.reduction_dim = dim // reduction_ratio
        self.dropout = dropout

        # 文本特征处理分支
        self.text_proj = nn.Sequential(
            nn.Linear(dim, self.reduction_dim),
            nn.LayerNorm(self.reduction_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 图像特征处理分支
        self.image_proj = nn.Sequential(
            nn.Linear(dim, self.reduction_dim),
            nn.LayerNorm(self.reduction_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 交叉注意力机制
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.reduction_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True  # 更直观的维度顺序
        )

        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(self.reduction_dim * 2, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid()
        )

        # 最终投影层
        self.final_proj = nn.Sequential(
            nn.Linear(self.reduction_dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """ Xavier/Glorot初始化 """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, text_feat, image_feat):
        # 残差连接的基础
        identity = (text_feat + image_feat) / 2

        # 降维投影
        text_reduced = self.text_proj(text_feat)  # [batch, reduction_dim]
        image_reduced = self.image_proj(image_feat)  # [batch, reduction_dim]

        # 交叉注意力 (query=text, key=value=image)
        attn_output, _ = self.cross_attn(
            query=text_reduced.unsqueeze(1),  # [batch, 1, reduction_dim]
            key=image_reduced.unsqueeze(1),
            value=image_reduced.unsqueeze(1)
        )
        attn_output = attn_output.squeeze(1)  # [batch, reduction_dim]

        # 拼接特征
        combined = torch.cat([attn_output, image_reduced], dim=-1)  # [batch, reduction_dim*2]

        # 门控融合
        gate = self.gate(combined)  # [batch, dim]

        # 最终投影 + 残差连接
        fused_feat = self.final_proj(combined)  # [batch, dim]
        output = gate * fused_feat + identity  # [batch, dim]

        return output

