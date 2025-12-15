import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Residual Block
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return x + out

# Simple Channel Attention (part of CBAM)
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        out = self.fc1(avg)
        out = self.relu(out)
        out = self.fc2(out)
        return torch.sigmoid(out) * x

# Simple Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,7,padding=3)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv,_ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, maxv], dim=1)
        attn = torch.sigmoid(self.conv(cat))
        return attn * x

class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# U-Net like encoder-decoder with residual blocks and CBAM
class DenoiseUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64):
        super().__init__()
        # encoder
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True), ResBlock(base), CBAMBlock(base))
        self.e2 = nn.Sequential(nn.Conv2d(base, base*2, 3, stride=2, padding=1), nn.ReLU(inplace=True), ResBlock(base*2), CBAMBlock(base*2))
        self.e3 = nn.Sequential(nn.Conv2d(base*2, base*4, 3, stride=2, padding=1), nn.ReLU(inplace=True), ResBlock(base*4), CBAMBlock(base*4))
        # bottleneck
        self.b = nn.Sequential(ResBlock(base*4), ResBlock(base*4))
        # decoder
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = nn.Sequential(ResBlock(base*2), CBAMBlock(base*2))
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = nn.Sequential(ResBlock(base), CBAMBlock(base))
        self.out = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        b = self.b(e3)
        u2 = self.up2(b)
        u2 = u2 + e2
        d2 = self.d2(u2)
        u1 = self.up1(d2)
        u1 = u1 + e1
        d1 = self.d1(u1)
        out = self.out(d1)
        return torch.clamp(out, 0.0, 1.0)
