import torch
from torch import nn as nn
from torch.nn.modules import utils
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights, make_layer

from .DFAB import DFAB

class BSConvU(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class CLKConvU(nn.Module):
    def __init__(self, in_channels=56, out_channels=56, kernel_size=5, stride=1, padding=2):
        super().__init__()

        assert in_channels==out_channels, "The input channel must equal the output channel"

        self.pw5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, groups=in_channels, dilation=kernel_size//2, padding=padding)
        self.pw3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, groups=out_channels, padding=1)
        self.pd1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0)
    
    def forward(self, x):
        out = self.pw5(x)
        out = self.pw3(out)
        out = self.pd1(out)
        out = out + x
        

        return out

class PartialBSConvU(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        scale=2,
    ):
        super().__init__()

        # pointwise
        self.remaining_channels = in_channels // scale
        self.other_channels = in_channels - self.remaining_channels
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # partialdepthwise
        self.pdw = nn.Conv2d(
            in_channels=self.remaining_channels,
            out_channels=self.remaining_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels // scale,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea1, fea2 = torch.split(
            fea, [self.remaining_channels, self.other_channels], dim=1
        )
        fea1 = self.pdw(fea1)
        fea = torch.cat((fea1, fea2), 1)
        fea = self.pw(fea)
        return fea

class EPartialBSConvU(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros",
                 scale=2,
                 status="train"):
        super().__init__()

        conv_ch = in_channels // scale

        self.rc = conv_ch
        self.oc = in_channels-conv_ch
        self.status = status

        self.kernel_size =kernel_size
        self.stride = stride
        self.bias = bias

        if status == "test":
            self.pw = nn.Conv2d(in_channels=conv_ch,
                                out_channels=conv_ch,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=kernel_size//2,
                                dilation=1,
                                bias=bias,
                                groups=self.rc)
        else:
            self.pws = nn.ModuleList()
            for i in range(1, kernel_size+1, 2):
                conv = nn.Conv2d(in_channels=conv_ch,
                                out_channels=conv_ch,
                                kernel_size=i,
                                stride=stride,
                                padding=i//2,
                                dilation=1,
                                bias=bias,
                                groups=self.rc)
                self.pws.append(conv)

        self.pd = nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                            bias=False)

    def forward(self, x):
        # w = torch.cat([self.w1, self.w2], dim=0)
        x1, x2 = torch.split(x, [self.rc, self.oc], dim=1)
        x1_rep = x1

        if self.status == "test":
            x1 = self.pw(x1)
        else:
            for conv in self.pws:
                x1 = x1 + conv(x1_rep)
        x = torch.cat([x1, x2], dim=1)
        
        x = self.pd(x)

        return x
    
    def switch_deploy(self,):
        kernnel, bias = self.get_equivalent_kernel_bias()
        self.pw = nn.Conv2d(in_channels=self.rc,
                                out_channels=self.rc,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                padding=self.kernel_size//2,
                                dilation=1,
                                bias=self.bias,
                                groups=self.rc)
        self.pw.weight.data = kernnel
        self.pw.bias.data = bias

        self.__delattr__('pws')
        self.status = "test"
        

    def get_equivalent_kernel_bias(self,):
        
        kernnel = torch.zeros(self.rc, 1, self.kernel_size, self.kernel_size)
        center = self.kernel_size // 2
        kernnel[:, :, center, center] = 1
        bias = torch.zeros(self.rc)

        for conv in self.pws:
            w = conv.weight.data
            padding = ((self.kernel_size-w.shape[-2])//2, (self.kernel_size-w.shape[-2])//2 , (self.kernel_size-w.shape[-1])//2, (self.kernel_size-w.shape[-1])//2)
            w = F.pad(w, padding)
            kernnel = kernnel + w
            bias = bias + conv.bias.data
        return kernnel, bias

class FSAB(nn.Module):
    def __init__(self, in_ch, out_ch, conv=nn.Conv2d, channel_reduction_rate=1, bias=True):
        super().__init__()
        self.dc = in_ch // 2
        self.rc = in_ch

        self.c1d = nn.Conv2d(in_channels=in_ch, out_channels=self.dc, kernel_size=1)
        self.c1r = conv(in_channels=in_ch, out_channels=self.rc, kernel_size=5, stride=1, padding=2)
        
        self.c2d = nn.Conv2d(in_channels=self.rc, out_channels=self.dc, kernel_size=1)
        self.c2r = conv(in_channels=self.rc, out_channels=self.rc, kernel_size=5, stride=1, padding=2)

        self.c3d = nn.Conv2d(in_channels=self.rc, out_channels=self.dc, kernel_size=1)
        self.c3r = conv(in_channels=self.rc, out_channels=self.rc, kernel_size=5, stride=1, padding=6, dilation=3)

        self.c4 = BSConvU(self.rc, self.dc, kernel_size=3, padding=1)

        self.c5 = nn.Conv2d(self.dc*4, out_channels=in_ch, kernel_size=1, stride=1, padding=0)

        self.dfab = DFAB(in_channels=in_ch, channel_reduction_rate=channel_reduction_rate, bias=bias)

        self.c6 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)

        self.act = nn.GELU()

        self.norm = nn.LayerNorm(out_ch)

        default_init_weights([self.norm], 0.1)

    def forward(self, x):
        x_c1d = self.act(self.c1d(x))
        x_c1r = self.act(self.c1r(x))

        x_c2d = self.act(self.c2d(x_c1r))
        x_c2r = self.act(self.c2r(x_c1r))

        x_c3d = self.act(self.c3d(x_c2r))
        x_c3r = self.act(self.c3r(x_c2r))

        x_c4 = self.act(self.c4(x_c3r))

        x_c4 = torch.cat([x_c1d, x_c2d, x_c3d, x_c4], dim=1)
        out = self.c5(x_c4)
        out = self.dfab(out)
        out = self.c6(out)

        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2).contiguous() 



        return x + out

def UpsampleOneStep(in_channels, out_channels, upscale_factor=4):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class Upsampler_rep(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels * 2, 1)
        self.conv3x3 = nn.Conv2d(in_channels * 2, out_channels * (upscale_factor**2), 3)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        v1 = F.conv2d(x, self.conv1x1.weight, self.conv1x1.bias, padding=0)
        v1 = F.pad(v1, (1, 1, 1, 1), "constant", 0)
        b0_pad = self.conv1x1.bias.view(1, -1, 1, 1)
        v1[:, :, 0:1, :] = b0_pad
        v1[:, :, -1:, :] = b0_pad
        v1[:, :, :, 0:1] = b0_pad
        v1[:, :, :, -1:] = b0_pad
        v2 = F.conv2d(v1, self.conv3x3.weight, self.conv3x3.bias, padding=0)
        out = self.conv1(x) + self.conv3(x) + v2
        return self.pixel_shuffle(out)

# @ARCH_REGISTRY.register()
class FSANet(nn.Module):
    def __init__(self, 
                 in_ch=3,
                 out_ch=3,
                 fea_ch=56,
                 blocks=8,
                 num_in=4,
                 upscale=4,
                 upsampler="pixelshuffledirect",
                 conv="BSConvU",
                 rgb_mean=None,
                #  rgb_mean=(0.4488, 0.4371, 0.4040),
                 channel_reduction_rate=1,
                 bias=True,):
        super().__init__()

        self.num_in = num_in
        self.blocks = blocks
        if rgb_mean is not None:
            self.rgb_mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        if conv == "BSConvU":
            self.conv = BSConvU
        elif conv == "CLKConvU":
            self.conv = CLKConvU
        elif conv == "PartialBSConvU":
            self.conv = PartialBSConvU
        elif conv == "EPartialBSConvU":
            self.conv = EPartialBSConvU
        else:
            raise NotImplementedError(f"conv {conv} is not supported yet.")
        
        self.first_conv = BSConvU(in_channels=in_ch*num_in, out_channels=fea_ch, kernel_size=3, stride=1, padding=1)

        self.body = nn.ModuleList()
        for _ in range(blocks):
            self.body.append(FSAB(in_ch=fea_ch, out_ch=fea_ch, conv=self.conv, channel_reduction_rate=channel_reduction_rate, bias=bias))
        
        self.c1 = nn.Conv2d(fea_ch * blocks, fea_ch, 1, 1, 0)
        self.c2 = BSConvU(fea_ch, fea_ch, kernel_size=3, padding=1)

        self.GELU = nn.GELU()

        if upsampler == "pixelshuffledirect":
            self.upsampler = UpsampleOneStep(
                fea_ch, out_ch, upscale_factor=upscale
            )
        elif upsampler == "pixelshuffle_rep":
            self.upsampler = Upsampler_rep(fea_ch, out_ch, upscale_factor=upscale)
        else:
            raise NotImplementedError("Check the Upsampler. None or not support yet.")

    def forward(self, x):
        if self.rgb_mean is not None:
            self.rgb_mean = self.rgb_mean.type_as(x)
            x = x - self.rgb_mean
        x = torch.cat([x] * self.num_in, dim=1)
        fea_x = self.first_conv(x)
        res_x = fea_x
        out_x = []

        for block in self.body:
            fea_x = block(fea_x)
            out_x.append(fea_x)

        out_x = torch.cat(out_x, dim=1)
        out_x = self.c1(out_x)
        out_x = self.GELU(out_x)
        out_x = self.c2(out_x) + res_x

        out_x = self.upsampler(out_x)

        if self.rgb_mean is not None:
            out_x = out_x + self.rgb_mean

        return out_x


if __name__ == "__main__":
    # model = CLKConvU()
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"CLKConvU parameters: {total_params}")

    # print("卷积参数量对比")
    # c = nn.Conv2d(in_channels=56, out_channels=56, kernel_size=3, stride=1, groups=56, dilation=3//2, padding=2)
    # total_params = sum(p.numel() for p in c.parameters())
    # print(f"c1 parameters: {total_params}")

    # c2 = nn.Conv2d(
    #         in_channels=56//2,
    #         out_channels=56//2,
    #         kernel_size=5,
    #         stride=1,
    #         padding=2,
    #         dilation=1,
    #         groups=56 // 2,
    #         bias=True,
    #         padding_mode="zeros",
    #     )
    # total_params = sum(p.numel() for p in c2.parameters())
    # print(f"c2 parameters: {total_params}")

    # model = FSANet(conv="CLKConvU", rgb_mean=[0.4488, 0.4371, 0.4040], bias=False, channel_reduction_rate=2)
    # for name, module in model.named_modules():
    #     if not name:  # 跳过根模块（即模型本身）
    #         continue
    #     num_params = sum(p.numel() for p in module.parameters())
    #     print(f"{name}: {num_params}")
    # for p in model.parameters():
    #     print(p.numel())
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"FSANet parameters: {total_params}")
    # x = torch.randn(64, 56, 64, 64)
    # y = model(x)
    # print(y.shape)

    # x = torch.randn(64, 56, 64, 64)  # 输入
    # conv = nn.Conv2d(56, 56, kernel_size=3, stride=1, padding=1, groups=56)
    # print(conv.weight.data.shape, " and ", conv.bias.data.shape)
    
    model = EPartialBSConvU(56, 56, 5, 1, 2, 1, True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"FSANet befor rep parameters: {total_params}")
    model.switch_deploy()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"FSANet after rep parameters: {total_params}")
    x = torch.randn(64, 56, 64, 64)
    y = model(x)
    print(y.shape)

    # model = PartialBSConvU(56, 56, 5, 1, 2, 1, True)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"PartialBSConvU parameters: {total_params}")

    # # 创建 CUDA 事件
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)

    # # 测试运行时间
    # start_event.record()  # 记录开始时间
    # for _ in range(10):
    #     output = model(x)     # 运行模块
    # end_event.record()   # 记录结束时间

    # # 等待事件完成
    # torch.cuda.synchronize()

    # # 计算运行时间
    # elapsed_time = start_event.elapsed_time(end_event)  # 返回毫秒
    # print(f"Elapsed time: {elapsed_time:.6f} milliseconds")

    # w = torch.zeros(3, 1, 5, 5)
    # w[:, 0, 5//2, 5//2] = 1
    # print(w)