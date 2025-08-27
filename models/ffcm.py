import torch
import torch.nn as nn

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0,
            groups=self.groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.real(ffted).unsqueeze(-1)
        x_fft_imag = torch.imag(ffted).unsqueeze(-1)
        ffted = torch.cat([x_fft_real, x_fft_imag], dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(batch, -1, *ffted.size()[3:])

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view(batch, -1, 2, *ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        return output

class FFCM(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super(FFCM, self).__init__()
        self.dim = dim

        # 1D -> 2D 适配层
        self.to_2d = nn.Sequential(
            nn.Linear(dim, dim * 4),  # [B,768] -> [B,3072]
            nn.Dropout(dropout_rate),
            nn.Unflatten(1, (dim, 2, 2))  # [B,768,2,2]
        )

        # 核心处理层
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),  # [B,768,2,2] -> [B,1536,2,2]
            nn.GELU()
        )
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.mixer = FourierUnit(dim * 2, dim * 2)
        self.bn = nn.BatchNorm2d(dim * 2)
        self.relu = nn.ReLU(inplace=True)

        # 修正后的2D -> 1D适配层
        self.to_1d = nn.Sequential(
            nn.Flatten(),  # [B,1536,2,2] -> [B,1536 * 2 * 2=6144]
            nn.Dropout(dropout_rate),
            nn.Linear(6144, dim)  # 明确匹配实际维度
        )

    def forward(self, x):

        x = self.to_2d(x)          # [B,768] -> [B,768,2,2]
        x = self.conv_init(x)       # [B,768,2,2] -> [B,1536,2,2]
        x1, x2 = torch.split(x, self.dim, dim=1)  # 各[B,768,2,2]
        x1 = self.dw_conv_1(x1)     # [B,768,2,2]
        x2 = self.dw_conv_2(x2)     # [B,768,2,2]
        x = self.mixer(torch.cat([x1, x2], dim=1))  # [B,1536,2,2]
        x = self.relu(self.bn(x))   # [B,1536,2,2]
        x = self.to_1d(x)# [B,6144] -> [B,768]
        
        return x