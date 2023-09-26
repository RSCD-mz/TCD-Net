import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .init_weights import init_weights

'''
    Frequency attention
'''
def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, groups=group, bias=bias)

import math
import torch
import torch.nn as nn


def get_ld_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)


def get_dct_weights(width, height, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, height)

    # split channel for multi-spectral attention
    c_part = channel // len(fidx_u)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                val = get_ld_dct(t_x, u_x, width) * get_ld_dct(t_y, v_y, height)
                dct_weights[:, i * c_part: (i+1) * c_part, t_x, t_y] = val

    return dct_weights

class FcaLayer_ori(nn.Module):
    def __init__(self, channels, reduction=16):
        super(FcaLayer_ori, self).__init__()
        self.register_buffer("precomputed_dct_weights", get_dct_weights(...))
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,_,_ = x.size()
        y = torch.sum(x * self.pre_computed_dct_weights, dim=[2,3])
        y = self.fc(y).view(n,c,1,1)
        return x * y.expand_as(x)

class FcaLayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(FcaLayer, self).__init__()
        # self.register_buffer("precomputed_dct_weights", get_dct_weights(...))
        # self.dct_weight = dct_weight
        self.fc1 = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )

        # CA
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        c2wh = dict([(64,224), (128,112), (256,56),(512,28), (1024, 14),(16,224),(35,224),(131,112),(32,224),(3,224)]) #---------------------
        
        # c2wh = eval([(64,224), (128,112), (256,56),(512,28), (1024, 14),(16,224),(35,224),(131,112),(32,224),(3,224)]) #---------------------
        

        self.att = MultiSpectralAttentionLayer(channels, c2wh[channels], c2wh[channels],  reduction=reduction, freq_sel_method = 'top4')


    def forward(self, x):
        n,c,_,_ = x.size()
        # y = torch.sum(x * self.pre_computed_dct_weights, dim=[2,3])
        # y = torch.sum(x*self.dct_weight, dim=[2,3])
        Frequency = self.att(x)  #n,c,1,1
        # Frequency = self.fc1(y).view(n,c,1,1)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out) + Frequency

        return out#x * y.expand_as(x)

#
class FaLayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(FaLayer, self).__init__()
        # self.register_buffer("precomputed_dct_weights", get_dct_weights(...))
        # self.dct_weight = dct_weight
        # self.fc1 = nn.Sequential(
        #     nn.Linear(channels, channels//reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channels//reduction, channels, bias=False),
        #     nn.Sigmoid()
        # )

        # CA
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        # self.fc = nn.Sequential(
        #     nn.Conv2d(channels, channels // 16, 1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(channels // 16, channels, 1, bias=False))
        # self.sigmoid = nn.Sigmoid()

        # c2wh = dict([(64,224), (128,112), (256,56),(512,28), (1024, 14)]) #---------------------
        c2wh = dict([(64,224), (128,112), (256,56),(512,28), (1024, 14),(16,224),(35,224),(131,112),(32,224),(3,224)]) #------
        
        self.att = MultiSpectralAttentionLayer(channels, c2wh[channels], c2wh[channels],  reduction=reduction, freq_sel_method = 'top16')


    def forward(self, x):
        n,c,_,_ = x.size()
        # y = torch.sum(x * self.pre_computed_dct_weights, dim=[2,3])
        # y = torch.sum(x*self.dct_weight, dim=[2,3])
        Frequency = self.att(x)  #n,c,1,1
        # Frequency = self.fc1(y).view(n,c,1,1)
        # avg_out = self.fc(self.avg_pool(x))
        # max_out = self.fc(self.max_pool(x))
        # out = self.sigmoid(avg_out + max_out) + Frequency
        # out = Frequency


        return Frequency#out#x * y.expand_as(x)

#
class FaLayer_sum(nn.Module):
    def __init__(self, channels, reduction=16):
        super(FaLayer_sum, self).__init__()
        # self.register_buffer("precomputed_dct_weights", get_dct_weights(...))
        # self.dct_weight = dct_weight
        # self.fc1 = nn.Sequential(
        #     nn.Linear(channels, channels//reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channels//reduction, channels, bias=False),
        #     nn.Sigmoid()
        # )

        # CA
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        # self.fc = nn.Sequential(
        #     nn.Conv2d(channels, channels // 16, 1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(channels // 16, channels, 1, bias=False))
        # self.sigmoid = nn.Sigmoid()

        # c2wh = dict([(64,224), (128,112), (256,56),(512,28), (1024, 14)]) #---------------------
        c2wh = dict([(64,224), (128,112), (256,56),(512,28), (1024, 14),(16,224),(35,224),(131,112),(32,224),(3,224)]) #------
        
        # self.att = MultiSpectralAttentionLayer(channels, c2wh[channels], c2wh[channels],  reduction=reduction, freq_sel_method = 'top4')
        self.att = MultiSpectralAttentionLayer_sum(channels, c2wh[channels], c2wh[channels],  reduction=reduction, freq_sel_method = 'top16')


    def forward(self, x):
        n,c,_,_ = x.size()
        # y = torch.sum(x * self.pre_computed_dct_weights, dim=[2,3])
        # y = torch.sum(x*self.dct_weight, dim=[2,3])
        Frequency = self.att(x)  #n,c,1,1
        # Frequency = self.fc1(y).view(n,c,1,1)
        # avg_out = self.fc(self.avg_pool(x))
        # max_out = self.fc(self.max_pool(x))
        # out = self.sigmoid(avg_out + max_out) + Frequency
        # out = Frequency


        return Frequency#out#x * y.expand_as(x)

#
class MultiSpectralAttentionLayer_sum(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer_sum, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        # self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.dct_layer = MultiSpectralDCTLayer_sum(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)  #n,c
        y = self.fc(y).view(n, c, 1, 1)   #n,c,1,1
        return y#x * y.expand_as(x)

class MultiSpectralDCTLayer_sum(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer_sum, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init  #  # 对应于公式中的H,W,i,j,C
        self.register_buffer('weight', self.get_dct_filter_sum(height, width, mapper_x, mapper_y, channel))
                
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight #n,c,h,w   *    c,h,w    =    n,c,h,w

        result = torch.sum(x, dim=[2,3]) ## 在空间维度上求和
        return result  # n,c----batch,channel

    def build_filter(self, pos, freq, POS): # 对应i/j, h/w, H/W
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)  # 基函数公式的一半
        if freq == 0:
            return result # 对应gap的形式
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):  # 对应与H,W,i,j,C
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y) # 对于每一个BATCH都是相同的

        c_part = channel // len(mapper_x)  # 每一份的通道长度

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                    # # 将f的求和式展开，储存在一个图中，这里实际上是一个3维的，因为这些通道的权重相同，最后再乘以原图求和得到f
                        
        return dct_filter#c,h,w

    def get_dct_filter_sum(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):  # 对应与H,W,i,j,C
        # dct_filter = torch.zeros(channel, tile_size_x, tile_size_y) # 对于每一个BATCH都是相同的

        # c_part = channel // len(mapper_x)  # 每一份的通道长度
        dct = torch.zeros(len(mapper_x), tile_size_x, tile_size_y)

        # for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
        #     for t_x in range(tile_size_x):
        #         for t_y in range(tile_size_y):
        #             dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
        #             # # 将f的求和式展开，储存在一个图中，这里实际上是一个3维的，因为这些通道的权重相同，最后再乘以原图求和得到f
        # a=torch.zeros

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct[i: i+1, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)

                    # dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                    # # 将f的求和式展开，储存在一个图中，这里实际上是一个3维的，因为这些通道的权重相同，最后再乘以原图求和得到f
        
        multifreweight = torch.sum(dct, dim=0) ## 在空间维度上求和
        # max =
        # avg = 
        
                        
        return multifreweight#dct_filter#c,h,w



class FC_att(nn.Module):
    def __init__(self, in_planes, reduction=16, ratio=16):
        super(FC_att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        # c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)]) #---------------------
        c2wh = dict([(64,112), (128,56), (256,28) ,(512,14)]) #---------------------

        planes = in_planes
        self.attout = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')#----------------

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # Frequency = self.attout(x)
        # out = self.sigmoid(avg_out + max_out)*x + Frequency
        # return 
        return avg_out+max_out

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top4'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)  #n,c
        y = self.fc(y).view(n, c, 1, 1)   #n,c,1,1
        return y#x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init  #  # 对应于公式中的H,W,i,j,C
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        self.aaaaa=1
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight #n,c,h,w   *    c,h,w    =    n,c,h,w

        result = torch.sum(x, dim=[2,3]) ## 在空间维度上求和
        return result  # n,c----batch,channel

    def build_filter(self, pos, freq, POS): # 对应i/j, h/w, H/W
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)  # 基函数公式的一半
        if freq == 0:
            return result # 对应gap的形式
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):  # 对应与H,W,i,j,C
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y) # 对于每一个BATCH都是相同的

        c_part = channel // len(mapper_x)  # 每一份的通道长度

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                    # # 将f的求和式展开，储存在一个图中，这里实际上是一个3维的，因为这些通道的权重相同，最后再乘以原图求和得到f
                        
        return dct_filter#c,h,w


class FCA_Block(nn.Module): #frequency channel attention
    expansion = 4
    def __init__(self, s, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(FCA_Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        if s == 2:
            self.globalAvgPool = nn.AvgPool2d((112, 112), stride=1)  # (224, 300) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((112, 112), stride=1)
        elif s == 4:
            self.globalAvgPool = nn.AvgPool2d((56, 56), stride=1)  # (112, 150) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((56, 56), stride=1)
        elif s == 8:
            self.globalAvgPool = nn.AvgPool2d((28, 28), stride=1)    # (56, 75) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((28, 28), stride=1)
        elif s == 16:
            self.globalAvgPool = nn.AvgPool2d((14, 14), stride=1)    # (28, 37) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((14, 14), stride=1)
        # elif planes == 256:
        #     self.globalAvgPool = nn.AvgPool2d((14, 18), stride=1)    # (14, 18) for ISIC2018
        #     self.globalMaxPool = nn.MaxPool2d((14, 18), stride=1)

        self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        avg_att = out
        out = out * original_out
        # For global maximum pool
        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        max_att = out1
        out1 = out1 * original_out

        att_weight = avg_att + max_att
        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out#, att_weight

'''
    fused-mbconv

'''
# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class FMBConv(nn.Module):
    """
    MBconv(inp, oup, stride, expand_ratio, fused)
    MBconv(3, 64, 1, 64, 1)

     定义MBConv模块和Fused-MBConv模块，将fused设置为1或True是Fused-MBConv，否则是MBConv
    :param inp:输入的channel
    :param oup:输出的channel
    :param stride:步长，设置为1时图片的大小不变，设置为2时，图片的面积变为原来的四分之一
    :param expand_ratio:放大的倍率
    :return:
    """
    def __init__(self, inp, oup, stride):#, expand_ratio, fused):
        super(MBConv, self).__init__()
        assert stride in [1, 2]
        hidden_dim = 64#round(inp * expand_ratio)
        # self.identity = stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # fused
            nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            SiLU(),
            SELayer(inp, hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
            # nn.MaxPool2d(2)
            )

    def forward(self, x):
        return x + self.conv(x)

class FMBConv_down(nn.Module):
    """
    FMBConv_down(inp, oup, stride, expand_ratio, fused)
    FMBConv_down(3, 64, 1, 64, 1)

     定义MBConv模块和Fused-MBConv模块，将fused设置为1或True是Fused-MBConv，否则是MBConv
    :param inp:输入的channel
    :param oup:输出的channel
    :param stride:步长，设置为1时图片的大小不变，设置为2时，图片的面积变为原来的四分之一
    :param expand_ratio:放大的倍率
    :return:
    """
    def __init__(self, inp, oup, stride, expand_ratio, fused):
        super(FMBConv_down, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        # self.identity = stride == 1 and inp == oup

        self.down = nn.MaxPool2d(2)
        self.chconv = DeepWise_PointWise_Conv(inp, oup)

        self.conv = nn.Sequential(
            # fused
            # nn.MaxPool2d(2),
            nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            SiLU(),
            SELayer(inp, hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
            )
    def forward(self, x):
        x = self.down(x)
        add = self.chconv(x)
        y = self.conv(x)
        return add + y

class FMBConv_down_connect(nn.Module):
    """
    MBconv(s, inp, oup, stride, expand_ratio, fused)
    MBconv(3, 64, 1, 64, 1)

     定义MBConv模块和Fused-MBConv模块，将fused设置为1或True是Fused-MBConv，否则是MBConv
    :param inp:输入的channel
    :param oup:输出的channel
    :param stride:步长，设置为1时图片的大小不变，设置为2时，图片的面积变为原来的四分之一
    :param expand_ratio:放大的倍率
    :return:
    """
    def __init__(self, s, inp, oup, stride, expand_ratio, fused):
        super(FMBConv_down_connect, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        # self.identity = stride == 1 and inp == oup

        # self.down = nn.MaxPool2d(s, s, ceil_mode=True)
        self.down = nn.MaxPool2d(s, ceil_mode=True)

        self.chconv = DeepWise_PointWise_Conv(inp, oup)

        self.conv = nn.Sequential(
            # fused
            # nn.MaxPool2d(s, s, ceil_mode=True),
            nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            SiLU(),
            SELayer(inp, hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
            )
    def forward(self, x):
        x = self.down(x)
        add = self.chconv(x)
        y = self.conv(x)
        return add + y

class FMBConv_up_connect(nn.Module):
    """
    MBconv(inp, oup, stride, expand_ratio, fused)
    MBconv(3, 64, 1, 64, 1)

     定义MBConv模块和Fused-MBConv模块，将fused设置为1或True是Fused-MBConv，否则是MBConv
    :param inp:输入的channel
    :param oup:输出的channel
    :param stride:步长，设置为1时图片的大小不变，设置为2时，图片的面积变为原来的四分之一
    :param expand_ratio:放大的倍率
    :return:
    """
    def __init__(self, size, inp, oup, stride, expand_ratio, fused):
        super(FMBConv_up_connect, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        # self.identity = stride == 1 and inp == oup            
        self.up = nn.Upsample(size, mode='bilinear')
        self.chconv = DeepWise_PointWise_Conv(inp, oup)

        self.conv = nn.Sequential(
            # fused
            # nn.Upsample(size, mode='bilinear'),
            nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            SiLU(),
            SELayer(inp, hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
            )
    def forward(self, x):
        x = self.up(x)
        add = self.chconv(x)
        y = self.conv(x)
        return add + y

class FMBConv_concat(nn.Module):
    """
    MBconv(s, inp, oup, stride, expand_ratio, fused)
    MBconv(3, 64, 1, 64, 1)

     定义MBConv模块和Fused-MBConv模块，将fused设置为1或True是Fused-MBConv，否则是MBConv
    :param inp:输入的channel
    :param oup:输出的channel
    :param stride:步长，设置为1时图片的大小不变，设置为2时，图片的面积变为原来的四分之一
    :param expand_ratio:放大的倍率
    :return:
    """
    def __init__(self, inp, oup, stride, expand_ratio, fused):
        super(FMBConv_concat, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        # self.identity = stride == 1 and inp == oup
        self.chconv = DeepWise_PointWise_Conv(inp, oup)


        self.conv = nn.Sequential(
            # fused
            # nn.MaxPool2d(s, s, ceil_mode=True),
            nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            SiLU(),
            SELayer(inp, hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
            )
    def forward(self, x):
        add = self.chconv(x)
        return add + self.conv(x)

class MBConv(nn.Module):
    """
     定义MBConv模块和Fused-MBConv模块，将fused设置为1或True是Fused-MBConv，否则是MBConv
    :param inp:输入的channel
    :param oup:输出的channel
    :param stride:步长，设置为1时图片的大小不变，设置为2时，图片的面积变为原来的四分之一
    :param expand_ratio:放大的倍率
    :return:
    """
    def __init__(self, inp, oup, stride, expand_ratio, fused):
        super(MBConv, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if fused:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
 
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
 
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

'''
    CD_COmpression used
'''
class out(nn.Module):
    def __init__(self, in_ch, out_ch, s):
        super(out, self).__init__()
        self.mpconv = nn.Sequential(
            DeepWise_PointWise_Conv(in_ch, out_ch),
            nn.Upsample(s, mode='bilinear')
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class out1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(out1, self).__init__()
        self.mpconv = nn.Sequential(
            DeepWise_PointWise_Conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class depth_down_connect(nn.Module):
    def __init__(self, s, in_ch, out_ch):
        super(depth_down_connect, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(s, s, ceil_mode=True),
            DeepWise_PointWise_Conv(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class depth_up_connect(nn.Module):
    def __init__(self, s, in_ch, out_ch):
        super(depth_up_connect, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Upsample(s, mode='bilinear'),
            DeepWise_PointWise_Conv(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up_connect(nn.Module):
    def __init__(self, scale, in_ch, out_ch, bilinear=True):
        super(up_connect, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        # else:
        #     self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class depth_convcat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(depth_convcat, self).__init__()
        self.mpconv = nn.Sequential(
            DeepWise_PointWise_Conv(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class convcatup(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(depth_convcat, self).__init__()
        self.mpconv = nn.Sequential(
            DeepWise_PointWise_Conv(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class depth_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(depth_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            depth_double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class upconv(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(upconv, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class uppyramid(nn.Module):
    def __init__(self, in_ch,swin_ch, out_ch, bilinear=True):
        super(uppyramid, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch+swin_ch, out_ch)

    def forward(self, x1, x2, swin):
        x1 = self.up(torch.cat([swin, x1], dim=1))
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

#

class uppyramid_depth(nn.Module):
    def __init__(self, in_ch,swin_ch, out_ch, bilinear=True):
        super(uppyramid_depth, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        # self.conv = depth_double_conv(in_ch+swin_ch, out_ch)
        self.conv = depth_double_conv(in_ch+swin_ch+out_ch, out_ch)


    def forward(self, x1, x2, swin):
        x1 = self.up(torch.cat([swin, x1], dim=1))
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

#

# class up(nn.Module):
#     def __init__(self, in_ch, out_ch, bilinear=True):
#         super(up, self).__init__()

#         #  would be a nice idea if the upsampling could be learned too,
#         #  but my machine do not have enough memory to handle all those weights
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

#         self.conv = double_conv(in_ch, out_ch)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
        
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
#                         diffY // 2, diffY - diffY//2))
        
#         # for padding issues, see 
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x
# #

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class depth_double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(depth_double_conv, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch, 3, padding=1),
            DeepWise_PointWise_Conv(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, 3, padding=1),
            DeepWise_PointWise_Conv(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DeepWise_PointWise_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeepWise_PointWise_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
    
class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
