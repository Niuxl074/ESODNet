import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from einops import rearrange
from ..modules.conv import Conv, DWConv, RepConv, GhostConv, autopad
from ..modules.block import *
from .attention import *
from .rep_block import DiverseBranchBlock
from .kernel_warehouse import KWConv
from .dynamic_snake_conv import DySnakeConv
from .ops_dcnv3.modules import DCNv3, DCNv3_DyHead
from .shiftwise_conv import ReparamLargeKernelConv
from .mamba_vss import VSSBlock
from .fadc import AdaptiveDilatedConv
from ..backbone.repvit import Conv2d_BN, RepVGGDW, SqueezeExcite
from ..backbone.rmt import RetBlock, RelPos2d

from .orepa import *
from .RFAConv import *
from ultralytics.utils.torch_utils import make_divisible
from timm.layers import trunc_normal_
from timm.layers import CondConv2d

__all__ = [ 'ADown', 'FocusFeature', ]



class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)



######################################## RTM end ########################################

######################################## PKINet start ########################################

class GSiLU(nn.Module):
    """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""
    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))

class PKIModule_CAA(nn.Module):
    def __init__(self, ch, h_kernel_size = 11, v_kernel_size = 11) -> None:
        super().__init__()
        
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor

class PKIModule(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True) -> None:
        super().__init__()
        hidc = make_divisible(int(ouc * expansion), 8)
        
        self.pre_conv = Conv(inc, hidc)
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc, hidc, kernel_size=k, padding=autopad(k), groups=hidc) for k in kernel_sizes)
        self.pw_conv = Conv(hidc, hidc)
        self.post_conv = Conv(hidc, ouc)
        
        if with_caa:
            self.caa_factor = PKIModule_CAA(hidc, caa_kernel_size, caa_kernel_size)
        else:
            self.caa_factor = None
        
        self.add_identity = add_identity and inc == ouc
    
    def forward(self, x):
        x = self.pre_conv(x)
        
        y = x
        x = self.dw_conv[0](x)
        x = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv[1:]], dim=0), dim=0)
        x = self.pw_conv(x)
        
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x

class C3_PKIModule(C3):
    def __init__(self, c1, c2, n=1, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True, g=1, e=0.5):
        super().__init__(c1, c2, n, True, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(PKIModule(c_, c_, kernel_sizes, expansion, with_caa, caa_kernel_size, add_identity) for _ in range(n)))

class C2f_PKIModule(C2f):
    def __init__(self, c1, c2, n=1, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True, g=1, e=0.5):
        super().__init__(c1, c2, n, True, g, e)
        self.m = nn.ModuleList(PKIModule(self.c, self.c, kernel_sizes, expansion, with_caa, caa_kernel_size, add_identity) for _ in range(n))

class RepNCSPELAN4_CAA(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)
        self.caa = CAA(c3+(2*c4))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(self.caa(torch.cat(y, 1)))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(self.caa(torch.cat(y, 1)))

######################################## PKINet end ########################################

######################################## Focus Diffusion Pyramid Network end ########################################

class FocusFeature(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5) -> None:
        super().__init__()
        hidc = int(inc[1] * e)
        
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc[0], hidc, 1)
        )
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()
        self.conv3 = ADown(inc[2], hidc)
        
        
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)
        self.pw_conv = Conv(hidc * 3, hidc * 3)
    
    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        
        x = torch.cat([x1, x2, x3], dim=1)
        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv[1:]], dim=0), dim=0)
        feature = self.pw_conv(feature)
        
        x = x + feature
        return x
   
######################################## Focus Diffusion Pyramid Network end ########################################

######################################## Frequency-Adaptive Dilated Convolution start ########################################


