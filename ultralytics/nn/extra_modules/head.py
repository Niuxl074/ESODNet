import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ..modules import Conv, DFL, C2f, RepConv, Proto, Segment, Pose, OBB
from ..modules.conv import autopad
from .block import *
from .rep_block import *
from .afpn import AFPN_P345, AFPN_P345_Custom, AFPN_P2345, AFPN_P2345_Custom
from .dyhead_prune import DyHeadBlock_Prune
from .block import DyDCNv2
from ultralytics.utils.tal import dist2bbox, make_anchors, dist2rbox

__all__ = [ 'Detect_LSCD', ]





class Detect_LSCD(nn.Module):
    # Lightweight Shared Convolutional Detection Head
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction  是否动态调整网格重建
    export = False  # export mode        是否启用导出模式
    shape = None                        #图像形状
    anchors = torch.empty(0)  # init    初始化  anchors
    strides = torch.empty(0)  # init    初始化  strides

    def __init__(self, nc=80, hidc=256, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes  分类数量
        self.nl = len(ch)  # number of detection layers  检测层的数量
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)  DFL通道数（根据不同模型的大小）
        self.no = nc + self.reg_max * 4  # number of outputs per anchor  每个anchor的输入数量
        self.stride = torch.zeros(self.nl)  # strides computed during build  初始化步长
        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 1)) for x in ch)  #每个检测层的卷积和归一化操作
        self.share_conv = nn.Sequential(Conv_GN(hidc, hidc, 3), Conv_GN(hidc, hidc, 3))  #
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)   #用于预测边界框
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)       #   用于预测类别
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)   #缩放操作
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()  #DFL操作

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])    #对每一层应用卷积
            x[i] = self.share_conv(x[i])   #共享卷积操作
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)  #拼接预测结果
        if self.training:  # Training path  训练模式
            return x

        # Inference path    推理模式
        shape = x[0].shape  #  获取输入形状 BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  #合并所有检测层的输出
        if self.dynamic or self.shape != shape:
            #动态或形状变化时重新计算anchors, strides
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            #导出模型时处理
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)  #分割边界框和类别
        dbox = self.decode_bboxes(box)  #解码边界框

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            # 在导出模式下，预计算归一化因子以增加数值稳定性
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)  #拼接解码后的边界框和类别概率
        return y if self.export else (y, x)  # 返回最终结果或训练数据
    

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability. 初始化detect（）的偏置"""
        m = self  # self.model[-1]  # Detect() module box当前模块
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        m.cv2.bias.data[:] = 1.0  # 设置边界框的偏置
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # 解码边界框cls (.01 objects, 80 classes, 640 img)  设置类别的偏置

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


