import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_nums):
    """
    Args:
        x: (B, C, H, W)
        window_nums (int): 窗口数量的开方

    Returns:
        windows: (B, C*window_nums**2, H//window_nums, W//window_nums)
    """
    B, C, H, W= x.shape

    H_window_size = H // window_nums
    W_window_size = W // window_nums
    if H % window_nums!= 0:
        H_window_size += 1
    if W % window_nums!= 0:
        W_window_size += 1

    H_pad = (H_window_size - H % H_window_size) % H_window_size
    W_pad = (W_window_size - W % W_window_size) % W_window_size
    x = F.pad(x, (0, W_pad, 0, H_pad), mode='constant', value=0)


    x = x.view(B, C, window_nums, H_window_size, window_nums, W_window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H_window_size, W_window_size)
    return windows, H_pad, W_pad


def window_reverse(windows, window_nums, H, W, H_pad, W_pad):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    H = H + H_pad
    W = W + W_pad

    C = int(windows.shape[1] / (window_nums ** 2))
    x = windows.view(-1, C, window_nums, window_nums, H // window_nums, W // window_nums)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, H, W)
    return x[:, :, :H-H_pad, :W-W_pad]


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        #使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class DFABlock(nn.Module):
    def __init__(self, in_channels, convs=1, window_nums=5, fft_norm="ortho", bias=True):
        super(DFABlock, self).__init__()

        self.in_channels = in_channels
        self.window_nums = window_nums
        self.fft_norm = fft_norm
        self.bias = bias

        # self.cam = ChannelAttentionModule(in_channels) 不使用通道注意力

        # 创建convs个卷积块
        self.convs = nn.ModuleList()
        for i in range(convs):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels*window_nums**2, in_channels*window_nums**2, kernel_size=1, padding=0, groups=in_channels*window_nums**2, bias=bias),
                # nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
        
        if not bias:
            self.biasw = nn.Parameter(torch.zeros(1, in_channels, 1, 1), requires_grad=True)

    
    def forward(self, x):
        x_w, H_pad, W_pad = window_partition(x, self.window_nums)
        for conv in self.convs:
            x_w = conv(x_w)
        x_w = window_reverse(x_w, self.window_nums, x.shape[-2], x.shape[-1], H_pad, W_pad)
        if not self.bias:
            x_w = x_w + self.biasw

        return x_w + x

class DFAB(nn.Module):
    def __init__(self, in_channels, window_nums=5, fft_norm="ortho", channel_reduction_rate=1, bias=True):
        super(DFAB, self).__init__()
        print("使用DFABgv2")
        self.in_channels = in_channels*window_nums**2
        # self.convI = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0, bias=False)
        # self.convO = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, padding=0, bias=False)

        self.conv1 = nn.Conv2d(in_channels, in_channels//channel_reduction_rate, kernel_size=1, padding=0) # 不减少通道数
        # self.conv2 = nn.Conv2d(self.in_channels//2, self.in_channels//2, kernel_size=1, padding=0, groups=self.in_channels//2)
        self.conv2 = DFABlock(in_channels//channel_reduction_rate, convs=1, window_nums=window_nums, fft_norm=fft_norm, bias=bias)
        self.conv3 = nn.Conv2d(in_channels//channel_reduction_rate, in_channels, kernel_size=1, padding=0)
        # self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm
        self.window_nums = window_nums

    def forward(self, x):
        '''
        输入形状为(B, C, H, W)
        '''
        fft_dim = (-2, -1)
        # x = self.convI(x)

        x_fft = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)

        x_fft_r = self.act(self.conv1(x_fft.real))
        x_fft_i = self.act(self.conv1(x_fft.imag))

        x_fft_r = self.act(self.conv2(x_fft_r))
        x_fft_i = self.act(self.conv2(x_fft_i))
        # x_fft_r = self.conv2(x_fft_r) # 激活函数后置
        # x_fft_i = self.conv2(x_fft_i)

        x_fft_r = self.conv3(x_fft_r) + x_fft.real
        x_fft_i = self.conv3(x_fft_i) + x_fft.imag


        x_fft = torch.complex(x_fft_r, x_fft_i)
        x_fft = torch.fft.irfftn(x_fft, s=x.shape[-2:], dim=fft_dim, norm=self.fft_norm)

        x = x * x_fft
        return x
    
class Attention(nn.Module):

    def __init__(self, embed_dim, fft_norm="ortho"):
        # bn_layer not used
        super(Attention, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.conv_layer2 = torch.nn.Conv2d(embed_dim // 2, embed_dim // 2, 1, 1, 0)
        self.conv_layer3 = torch.nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        real = ffted.real + self.conv_layer3(
            self.relu(self.conv_layer2(self.relu(self.conv_layer1(ffted.real))))
        )
        imag = ffted.imag + self.conv_layer3(
            self.relu(self.conv_layer2(self.relu(self.conv_layer1(ffted.imag))))
        )
        ffted = torch.complex(real, imag)

        ifft_shape_slice = x.shape[-2:]

        output = torch.fft.irfftn(
            ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm
        )

        return x * output

if __name__ == '__main__':
    model = DFAB(56, channel_reduction_rate=2, bias=False)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"DFAB parameters: {total_params}")

    attn = Attention(56)
    total_params = sum(p.numel() for p in attn.parameters())
    print(f"Attention parameters: {total_params}")

    DFAblock = nn.Conv2d(28*5**2, 28*5**2, kernel_size=1, padding=0, groups=28*5**2, bias=False)
    total_params = sum(p.numel() for p in DFAblock.parameters())
    print(f"DAFBlock parameters: {total_params}")

    conv = torch.nn.Conv2d(56 // 2, 56 // 2, 1, 1, 0)
    total_params = sum(p.numel() for p in conv.parameters())
    print(f"conv parameters: {total_params}")
    # x = torch.randn(3, 52, 4*252, 4*165)
    # y = model(x)
    # print(y.shape)
    # print(y.detype)
    # DFAblock = DFABlock()