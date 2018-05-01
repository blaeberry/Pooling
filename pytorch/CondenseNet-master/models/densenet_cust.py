from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from layers import Conv
from torch.nn.modules.utils import _pair
from torch.nn.modules.padding import ConstantPad3d

__all__ = ['DenseNetCust']


def make_divisible(x, y):
    return int((x // y + 1) * y) if x % y else int(x)

class mixgb(nn.Module):
    def __init__(self, args, stride=2, padding=0):
        super(mixgb, self).__init__()
        kernel_size = args.kernel_size
        self.kernel_size = _pair(args.kernel_size)
        self.stride = stride
        self.padding = _pair(padding)
        self.b = args.bgates
        self.nomax = args.nomax
        self.noavg = args.noavg

        if self.noavg:
            self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)
        elif self.nomax:
            self.avgpool = nn.AvgPool2d(kernel_size, stride, padding)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)
            self.avgpool = nn.AvgPool2d(kernel_size, stride, padding)
            self.mw = nn.Parameter(torch.Tensor(1))
            if self.b:
                self.aw = nn.Parameter(torch.Tensor(1))

    def __repr__(self):
        s = ('{name}(kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        #if self.bias is None:
        #    s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        if self.noavg:
            out = self.maxpool(x)
        elif self.nomax:
            out = self.avgpool(x)
        else:
            max_out = self.maxpool(x)
            avg_out = self.avgpool(x)
            max_w = self.mw
            if self.b:
                avg_w = self.aw
            else:
                max_w = F.sigmoid(max_w)
                avg_w = 1.0 - max_w
            out = (max_out*max_w)+(avg_out*avg_w)

        return out


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        ### 1x1 conv i --> b*k
        self.conv_1 = Conv(in_channels, args.bottleneck * growth_rate,
                           kernel_size=1, groups=self.group_1x1)
        ### 3x3 conv b*k --> k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, args)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(_Transition, self).__init__()
        padding = 0
        if args.kernel_size == 3:
            padding = 1

        if args.convs:
            if args.no1x1:
                # purely 3x3 conv stride 2
                self.conv = Conv(in_channels, out_channels, 
                    kernel_size=args.kernel_size, padding=padding, stride=2)
                self.pool = nn.Sequential()
            elif args.dw:
                # depthwise separable convolutions
                self.conv = Conv(in_channels, in_channels, kernel_size=args.kernel_size, 
                    padding=padding, groups=in_channels, stride=2)
                #self.pool = Conv(in_channels, out_channels, kernel_size=1)
                self.pool = nn.Conv2d(in_channels, out_channels,
                    kernel_size=1, bias=True) #adding bias because no BN before
            else:
                # 1x1 into 3x3 conv stride 2
                self.conv = Conv(in_channels, out_channels, 
                    kernel_size=1, padding=padding, groups=args.group_1x1)
                #self.pool = Conv(out_channels, out_channels, kernel_size=args.kernel_size, 
                #    padding=padding, stride=2)
                self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=args.kernel_size, 
                    padding=padding, stride=2, bias=True) #adding bias because no BN before
        elif args.dw:
            # 1x1 into 3x3 conv stride 2 dw
            self.conv = Conv(in_channels, out_channels, kernel_size=1)
            # self.pool = Conv(out_channels, out_channels, kernel_size=args.kernel_size, 
                # padding=padding, stride=2, groups=out_channels)
            self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=args.kernel_size, 
                    padding=padding, stride=2, bias=True, groups=out_channels) #adding bias because no BN before
        elif args.noavg and args.nomax:
            # 1x1 stride 2
            self.conv = Conv(in_channels, out_channels, kernel_size=1, stride=2)
            self.pool = nn.Sequential()
        else:
            self.conv = Conv(in_channels, out_channels,
                             kernel_size=1, groups=args.group_1x1)
            self.pool = mixgb(args, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNetCust(nn.Module):
    def __init__(self, args):

        super(DenseNetCust, self).__init__()

        self.stages = args.stages
        self.growth = args.growth
        self.reduction = args.reduction
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.data in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        self.features = nn.Sequential()
        ### Set initial width to 2 x growth_rate[0]
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i, args)
        ### Linear layer
        self.classifier = nn.Linear(self.num_features, args.num_classes)

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
            elif isinstance(m, mixgb):
                if (not m.noavg) and (not m.nomax):
                    m.mw.data.fill_(0.5)
                if m.b:
                    m.aw.data.fill_(0.5)


    def add_block(self, i, args):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            args=self.args
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            out_features = make_divisible(math.ceil(self.num_features * self.reduction),
                                          self.args.group_1x1)
            trans = _Transition(in_channels=self.num_features,
                                out_channels=out_features,
                                args=self.args)
            self.features.add_module('transition_%d' % (i + 1), trans)
            self.num_features = out_features
        else:
            self.features.add_module('norm_last',
                         nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            if args.dw and args.noavg:
                self.features.add_module('1x1_last', nn.Conv2d(self.num_features, self.num_features,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0, bias=False))
                self.features.add_module('dw_last', nn.Conv2d(self.num_features, self.num_features,
                                          kernel_size=self.pool_size,
                                          stride=1,
                                          padding=0, bias=True,
                                          groups=self.num_features))
            else:
                ### Use adaptive ave pool as global pool
                self.features.add_module('pool_last',
                                         nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out
