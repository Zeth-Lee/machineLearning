import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Residual Block
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return self.relu(out)

# Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ===== 改进的U-Net:添加了真正的skip connections =====
class DenoiseUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            ResBlock(base),
            CBAMBlock(base)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base*2),
            nn.ReLU(inplace=True),
            ResBlock(base*2),
            CBAMBlock(base*2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base*4),
            nn.ReLU(inplace=True),
            ResBlock(base*4),
            CBAMBlock(base*4)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(base*4),
            ResBlock(base*4)
        )
        
        # Decoder with proper skip connections
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base*4, base*2, 3, padding=1, bias=False),  # base*4 due to concat
            nn.BatchNorm2d(base*2),
            nn.ReLU(inplace=True),
            ResBlock(base*2),
            CBAMBlock(base*2)
        )
        
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base*2, base, 3, padding=1, bias=False),  # base*2 due to concat
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            ResBlock(base),
            CBAMBlock(base)
        )
        
        # Output
        self.out = nn.Conv2d(base, out_ch, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)      # 28x28
        e2 = self.enc2(e1)     # 14x14
        e3 = self.enc3(e2)     # 7x7
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decoder with skip connections
        d2 = self.up2(b)       # 14x14
        d2 = torch.cat([d2, e2], dim=1)  # Concatenate skip connection
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)      # 28x28
        d1 = torch.cat([d1, e1], dim=1)  # Concatenate skip connection
        d1 = self.dec1(d1)
        
        out = self.out(d1)
        return torch.clamp(out, 0.0, 1.0)


# ===== 轻量级替代方案:适合小图像 =====
class LightDenoiseNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=48):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 浅层特征提取
        self.body = nn.Sequential(
            ResBlock(base),
            CBAMBlock(base),
            ResBlock(base),
            CBAMBlock(base),
            ResBlock(base),
            CBAMBlock(base),
            ResBlock(base)
        )
        
        self.tail = nn.Conv2d(base, out_ch, 3, padding=1)
    
    def forward(self, x):
        residual = x
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        # 残差学习:学习噪声而不是干净图像
        out = residual - x
        return torch.clamp(out, 0.0, 1.0)