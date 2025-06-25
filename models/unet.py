"""
Attention UNet Architecture
---------------------------
Modified UNet with:
1. Attention gates in decoder
2. Residual blocks
3. Progressive scaling support

Key components:
- AttentionGate: Feature selection mechanism
- ResidualBlock: Skip-connection block
- UNet: Main model with configurable depth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """Attention gate for selective feature propagation"""
    def __init__(self, in_channels_g, in_channels_x, inter_channels):
        """
        Args:
            in_channels_g: Gating signal channels
            in_channels_x: Skip connection channels
            inter_channels: Intermediate channels
        """
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels_g, inter_channels, kernel_size=1),
            nn.InstanceNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels_x, inter_channels, kernel_size=1),
            nn.InstanceNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, g, x):
        """g: gating signal, x: skip connection"""
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # Attended features

class ResidualBlock(nn.Module):
    """Residual block with instance normalization"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_channels)
        
        # Shortcut connection for channel mismatch
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.InstanceNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += self.shortcut(residual)
        return self.relu(out)

class UNet(nn.Module):
    """Configurable UNet with attention and residual blocks
    
    Features:
    - depth: Controls network depth (4-7 typical)
    - base_channels: Base channel multiplier
    - Progressive scaling support via transfer_weights()
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, depth=5):
        super(UNet, self).__init__()
        self.depth = depth
        self.base_channels = base_channels
        
        if depth < 2:
            raise ValueError("Depth must be at least 2")
        
        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # First encoder
        self.encoders.append(ResidualBlock(in_channels, base_channels))
        self.pools.append(nn.MaxPool2d(2, ceil_mode=True))
        
        # Intermediate encoders
        channels = base_channels
        for _ in range(1, depth-1):
            next_channels = channels * 2
            self.encoders.append(ResidualBlock(channels, next_channels))
            self.pools.append(nn.MaxPool2d(2, ceil_mode=True))
            channels = next_channels
        
        # Bottleneck
        self.bottom = ResidualBlock(channels, channels*2)
        
        # Decoder path with attention
        self.upconvs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for _ in range(depth-1):
            self.upconvs.append(nn.Conv2d(channels*2, channels, 3, padding=1))
            self.attentions.append(AttentionGate(channels, channels, channels//2))
            self.decoders.append(ResidualBlock(channels*2, channels))
            channels //= 2  # Increase spatial resolution, decrease channels

        # Output convolution
        self.outconv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 1),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, x):
        # Encoder path
        skips = []
        for i in range(self.depth-1):
            x = self.encoders[i](x)
            skips.append(x)
            x = self.pools[i](x)
        
        x = self.bottom(x)
        
        # Decoder path with attention
        for i in range(self.depth-1):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = self.upconvs[i](x)
            att_skip = self.attentions[i](x, skips[-(i+1)])
            x = torch.cat([x, att_skip], dim=1)
            x = self.decoders[i](x)
        
        return self.outconv(x)
    
    def transfer_weights(self, pretrained_model):
        """Transfer weights from smaller model for progressive training"""
        # Encoder transfer
        min_depth = min(self.depth, pretrained_model.depth) - 1
        for i in range(min_depth):
            self.encoders[i].load_state_dict(pretrained_model.encoders[i].state_dict())
        
        # Decoder transfer (reverse order)
        for i in range(min_depth):
            src_idx = pretrained_model.depth - 2 - i
            dst_idx = self.depth - 2 - i
            self.upconvs[dst_idx].load_state_dict(pretrained_model.upconvs[src_idx].state_dict())
            self.attentions[dst_idx].load_state_dict(pretrained_model.attentions[src_idx].state_dict())
            self.decoders[dst_idx].load_state_dict(pretrained_model.decoders[src_idx].state_dict())
        
        # Bottleneck transfer
        if self.depth == pretrained_model.depth:
            self.bottom.load_state_dict(pretrained_model.bottom.state_dict())
    
    @staticmethod
    def init_weights(m):
        """Kaiming initialization for convolutional layers"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# Example progressive training workflow:
# 1. Train 256px model
model_256 = UNet(depth=5)
# 
# 2. Scale to 512px
# model_512 = UNet(depth=6)
# model_512.transfer_weights(model_256)
# 
# 3. Scale to 1024px
# model_1024 = UNet(depth=7)
# model_1024.transfer_weights(model_512)