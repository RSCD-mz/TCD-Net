import torch
import torch.nn as nn
from .layers import*


class InputInjection(nn.Module):
    """Downsampling module for CGNet."""

    def __init__(self, num_downsampling):
        super(InputInjection, self).__init__()
        self.pool = nn.ModuleList()
        for i in range(num_downsampling):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x

def build_norm_layer(ch):
    layer = nn.BatchNorm2d(ch, eps=0.01)
    for param in layer.parameters():
        param.requires_grad = True
    return layer


class GlobalContextExtractor(nn.Module):
    """Global Context Extractor for CGNet.

    This class is employed to refine the joint feature of both local feature
    and surrounding context.

    Args:
        channel (int): Number of input feature channels.
        reduction (int): Reductions for global context extractor. Default: 16.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self, channel, reduction=16, with_cp=False):
        super(GlobalContextExtractor, self).__init__()
        self.channel = channel
        self.reduction = reduction
        assert reduction >= 1 and channel >= reduction
        self.with_cp = with_cp
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        def _inner_forward(x):
            num_batch, num_channel = x.size()[:2]
            y = self.avg_pool(x).view(num_batch, num_channel)
            y = self.fc(y).view(num_batch, num_channel, 1, 1)
            return x * y

        return _inner_forward(x)

#
class FrequencyGuidedBlock(nn.Module):
    """Context Guided Block for CGNet.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation=2,
                 reduction=16,
                 skip_connect=True,
                 downsample=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='PReLU'),
                 with_cp=False):
        super(FrequencyGuidedBlock, self).__init__()
        self.with_cp = with_cp
        self.downsample = downsample

        # channels = out_channels if downsample else out_channels // 2
        channels = out_channels // 2
        if 'type' in act_cfg and act_cfg['type'] == 'PReLU':
            act_cfg['num_parameters'] = channels
        kernel_size = 3 if downsample else 1
        stride = 2 if downsample else 1
        padding = (kernel_size - 1) // 2
        # self.channel_shuffle = ChannelShuffle(2 if in_channels==in_channels//2*2 else in_channels)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            build_norm_layer(channels),
            nn.PReLU(num_parameters=channels)
        )

        self.f_loc = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, groups=channels, bias=False)

        self.f_sur = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation,
                               dilation=dilation, groups=channels, bias=False)

        self.bn = build_norm_layer(2 * channels)
        self.activate = nn.PReLU(2 * channels)

        # original bottleneck in CGNet: A light weight context guided network for segmantic segmentation
        # is removed for saving computation amount
        # if downsample:
        #     self.bottleneck = build_conv_layer(
        #         conv_cfg,
        #         2 * channels,
        #         out_channels,
        #         kernel_size=1,
        #         bias=False)

        self.skip_connect = skip_connect and not downsample

        self.f_glo = GlobalContextExtractor(out_channels, reduction, with_cp)
        # self.f_glo = CoordAtt(out_channels,out_channels,groups=reduction)

        self.frequency = FaLayer(2*channels) #------64

    def forward(self, x):

        def _inner_forward(x):
            # x = self.channel_shuffle(x)
            out = self.conv1x1(x) #n,c,h,w
            loc = self.f_loc(out) #n,c,h,w
            sur = self.f_sur(out) #n,c,h,w

            joi = torch.cat([loc, sur], 1)  # the joint feature

            join = joi + joi*self.frequency(joi)

            joi_feat = self.bn(join)
            joi_feat = self.activate(joi_feat)
            if self.downsample:
                pass
                # joi_feat = self.bottleneck(joi_feat)  # channel = out_channels
            # f_glo is employed to refine the joint feature
            out = self.f_glo(joi_feat)

            if self.skip_connect:
                return join + out
            else:
                return out

            # if self.skip_connect:
            #     return x + out
            # else:
            #     return out

        return _inner_forward(x)

class ContextGuidedBlock(nn.Module):
    """Context Guided Block for CGNet.

    This class consists of four components: local feature extractor,
    surrounding feature extractor, joint feature extractor and global
    context extractor.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        dilation (int): Dilation rate for surrounding context extractor.
            Default: 2.
        reduction (int): Reduction for global context extractor. Default: 16.
        skip_connect (bool): Add input to output or not. Default: True.
        downsample (bool): Downsample the input to 1/2 or not. Default: False.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='PReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation=2,
                 reduction=16,
                 skip_connect=True,
                 downsample=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='PReLU'),
                 with_cp=False):
        super(ContextGuidedBlock, self).__init__()
        self.with_cp = with_cp
        self.downsample = downsample

        # channels = out_channels if downsample else out_channels // 2
        channels = out_channels // 2
        if 'type' in act_cfg and act_cfg['type'] == 'PReLU':
            act_cfg['num_parameters'] = channels
        kernel_size = 3 if downsample else 1
        stride = 2 if downsample else 1
        padding = (kernel_size - 1) // 2
        # self.channel_shuffle = ChannelShuffle(2 if in_channels==in_channels//2*2 else in_channels)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            build_norm_layer(channels),
            nn.PReLU(num_parameters=channels)
        )

        self.f_loc = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, groups=channels, bias=False)

        self.f_sur = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation,
                               dilation=dilation, groups=channels, bias=False)

        self.bn = build_norm_layer(2 * channels)
        self.activate = nn.PReLU(2 * channels)

        # original bottleneck in CGNet: A light weight context guided network for segmantic segmentation
        # is removed for saving computation amount
        # if downsample:
        #     self.bottleneck = build_conv_layer(
        #         conv_cfg,
        #         2 * channels,
        #         out_channels,
        #         kernel_size=1,
        #         bias=False)

        self.skip_connect = skip_connect and not downsample
        self.f_glo = GlobalContextExtractor(out_channels, reduction, with_cp)
        # self.f_glo = CoordAtt(out_channels,out_channels,groups=reduction)

    def forward(self, x):

        def _inner_forward(x):
            # x = self.channel_shuffle(x)
            out = self.conv1x1(x)
            loc = self.f_loc(out)
            sur = self.f_sur(out)

            joi_feat = torch.cat([loc, sur], 1)  # the joint feature
            joi_feat = self.bn(joi_feat)
            joi_feat = self.activate(joi_feat)
            if self.downsample:
                pass
                # joi_feat = self.bottleneck(joi_feat)  # channel = out_channels
            # f_glo is employed to refine the joint feature
            out = self.f_glo(joi_feat)

            if self.skip_connect:
                return x + out
            else:
                return out

        return _inner_forward(x)


def cgblock(in_ch, out_ch, dilation=2, reduction=8, skip_connect=False):
    return nn.Sequential(
        ContextGuidedBlock(in_ch, out_ch,
                           dilation=dilation,
                           reduction=reduction,
                           downsample=False,
                           skip_connect=skip_connect))




class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, h=None, w=None, groups=32):
        super(CoordAtt, self).__init__()


        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self,x):
        self.pool_h = nn.AdaptiveAvgPool2d((x.shape[-2], 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, x.shape[-1]))
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x) # n,c,h,1
        x_w = self.pool_w(x).permute(0,1,3,2) # n,c,w,1

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y