import torch
import torch.nn as nn

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # Adaptive average pooling and max pooling to get channel-wise statistics
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Fully connected layers to learn channel attention
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),  # Reduce channels
            nn.ReLU(),  # Activation
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)  # Restore channels
        )
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation to get attention map

    def forward(self, x):
        # Apply average pooling and max pooling
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # Sum the outputs and apply sigmoid to get the final attention map
        out = avg_out + max_out
        return self.sigmoid(out)

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Convolution layer to learn spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation to get attention map

    def forward(self, x):
        # Compute average and max pooling along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along the channel dimension
        out = torch.cat([avg_out, max_out], dim=1)
        # Apply convolution and sigmoid to get the final attention map
        out = self.conv(out)
        return self.sigmoid(out)

# Convolutional Block Attention Module (CBAM)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Initialize channel and spatial attention modules
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_attention(x)
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        return x