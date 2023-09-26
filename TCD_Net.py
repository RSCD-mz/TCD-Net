# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import * #unetConv2
from .init_weights import init_weights
from .swin_block import*
from .lightmodel import*

# from layers import unetConv2
# from init_weights import init_weights
class backbone(nn.Module): ## 
    def __init__(self, n_channels, n_classes):
        super(backbone, self).__init__()

        self.inc = inconv(n_channels, 64)
        self.inc_p1 = inconv(n_channels, 64)

        self.down1 = depth_down(64, 128)
        self.down2 = depth_down(128, 256)
        self.down3 = depth_down(256, 512)
        self.down4 = depth_down(512, 512) 

        self.down_p1 = depth_down(64, 128)
        self.down_p2 = depth_down(128, 256)
        self.down_p3 = depth_down(256, 512) 

        self.up1 = upconv(1024, 256)
        self.up2 = upconv(512, 128)
        self.up3 = upconv(256, 64)
        self.up4 = upconv(128, 64)

        self.outc = outconv(64, n_classes)

        self.sigmoid = nn.Sigmoid()

        size = [224,112,56,28,14]

        self.diff_down1 = depth_down(64, 128)
        self.diff_down2 = depth_down(128, 256)
        self.diff_down3 = depth_down(256, 512)
        self.diff_down4 = depth_down(512, 512)

        self.diff2_conv = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.diff3_conv = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.diff4_conv = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.diff5_conv = nn.Conv2d(1024, 512, kernel_size=1, stride=1)

    def forward(self, x, m):  ## x pre path; m, post path
        x1 = self.inc(x)
        m1 = self.inc_p1(m)
        d1 = m1-x1 # 64,224,224
        
        x2 = self.down1(x1)
        m2 = self.down_p1(m1)
        d2 = m2-x2

        x3 = self.down2(x2)
        m3 = self.down_p2(m2)
        d3 = m3-x3

        x4 = self.down3(x3)
        m4 = self.down_p3(m3)
        d4 = m4-x4

        m5 = self.down4(d4)
        d5 = m5
        # diff5 = self.diff5_conv(torch.cat((x5,feature4), dim=1))

        x = self.up1(d5, d4)#
        x = self.up2(x, d3)
        x = self.up3(x, d2)
        x = self.up4(x, d1)
        x = self.outc(x)
            
        x = self.sigmoid(x)
        return x

#

class TCD_Net(nn.Module): ## 64,32----128,64--DIM=96,CI=128
    def __init__(self, n_channels, n_classes):
        super(TCD_Net, self).__init__()

        self.inc = inconv(n_channels, 64)
        self.inc_p1 = inconv(n_channels, 64)

        self.down1 = depth_down(64, 128)
        self.down2 = depth_down(128, 256)
        self.down3 = depth_down(256, 512)
        self.down4 = depth_down(512, 512) 

        self.down_p1 = depth_down(64, 128)
        self.down_p2 = depth_down(128, 256)
        self.down_p3 = depth_down(256, 512) 

        # self.decoder1_up1 = uppyramid(1024, 64, 256)
        # self.decoder1_up2 = uppyramid(512, 64,128)
        # self.decoder1_up3 = uppyramid(256, 64, 64)
        # self.decoder1_up4 = uppyramid(128, 64, 64)

        # self.decoder2_up1 = uppyramid_depth(256, 64,128)
        # self.decoder2_up2 = uppyramid_depth(128, 64, 64)
        # self.decoder2_up3 = uppyramid_depth(64, 64, 64)

        # self.decoder3_up1 = uppyramid_depth(128, 64, 64)
        # self.decoder3_up2 = uppyramid_depth(64, 64, 64)
        
        # self.decoder4_up1 = uppyramid_depth(64, 64, 64)

        self.outc = outconv(64, n_classes)

        self.sigmoid = nn.Sigmoid()

        filters = [64, 128, 256, 512, 1024]
        # size = [224, 112, 56, 28, 14]
        # p_size = [0, 1, 2, 4]

        self.FCA1 = FaLayer_sum(filters[0]) #------64
        self.FCA2 = FaLayer_sum(filters[1]) #------128
        self.FCA3 = FaLayer_sum(filters[2]) #------256
        self.FCA4 = FaLayer_sum(filters[3]) #------512

        self.coattpre1 = FaLayer_sum(filters[0]) #------64
        self.coattpre2 = FaLayer_sum(filters[1]) #------128
        self.coattpre3 = FaLayer_sum(filters[2]) #------256
        self.coattpre4 = FaLayer_sum(filters[3]) #------512

        self.coattpost1 = FaLayer_sum(filters[0]) #------64
        self.coattpost2 = FaLayer_sum(filters[1]) #------128
        self.coattpost3 = FaLayer_sum(filters[2]) #------256
        self.coattpost4 = FaLayer_sum(filters[3]) #------512

        # self.swin1 = Swin_Model(chan=3, dim=32, patch_size=p_size[1], num_heads=4, 
        #                     input_res=[224, 224], window_size=7, qkv_bias=True)

        p_size = [0, 1, 2, 4, 8]

        self.swin1 = Swin_Model(chan=64, dim=96, patch_size=p_size[1], num_heads=4,
                            input_res=[224, 224], window_size=7, qkv_bias=True)
        self.swin2 = Swin_Model(chan=128, dim=96, patch_size=p_size[1], num_heads=4, 
                            input_res=[112, 112], window_size=7, qkv_bias=True)
        self.swin3 = Swin_Model(chan=128, dim=96, patch_size=p_size[1], num_heads=4, 
                            input_res=[56, 56], window_size=7, qkv_bias=True)
        self.swin4 = Swin_Model(chan=128, dim=96, patch_size=p_size[1], num_heads=4, 
                            input_res=[28, 28], window_size=7, qkv_bias=True)

        #----------------------
        self.swin13 = Swin_Model(chan=64, dim=96, patch_size=2, num_heads=4, 
                            input_res=[112, 112], window_size=7, qkv_bias=True)
        self.swin14 = Swin_Model(chan=64, dim=96, patch_size=4, num_heads=4, 
                            input_res=[56, 56], window_size=7, qkv_bias=True)
        #***************
        self.swin15 = Swin_Model(chan=64, dim=96, patch_size=8, num_heads=4, 
                            input_res=[28, 28], window_size=7, qkv_bias=True)
        #****************

        self.swin24 = Swin_Model(chan=192, dim=96, patch_size=2, num_heads=4, 
                            input_res=[56, 56], window_size=7, qkv_bias=True)
        #**************
        self.swin25 = Swin_Model(chan=192, dim=96, patch_size=4, num_heads=4, 
                            input_res=[28, 28], window_size=7, qkv_bias=True)

        self.swin35 = Swin_Model(chan=192, dim=96, patch_size=2, num_heads=4, 
                            input_res=[28, 28], window_size=7, qkv_bias=True)
        #***************
        #*************
        # self.swin4 = Swin_Model(chan=32, dim=32, patch_size=p_size[1], num_heads=4, 
        #                     input_res=[28, 28], window_size=7, qkv_bias=True)
                            
        self.swconv1 = nn.Conv2d(filters[1]+1*192, 128, kernel_size=1, stride=1)
        self.swconv2 = nn.Conv2d(filters[2]+2*192, 128, kernel_size=1, stride=1)
        self.swconv3 = nn.Conv2d(filters[3]+3*192, 128, kernel_size=1, stride=1)
        self.swconv4 = nn.Conv2d(192*4, 128, kernel_size=1, stride=1)

        #------------------
        self.decoder1_up1 = uppyramid(512*2, 128, 256)#in,swin,out
        self.decoder1_up2 = uppyramid(256*2, 192,128)
        self.decoder1_up3 = uppyramid(128*2, 192, 64)
        self.decoder1_up4 = uppyramid(64*2, 192, 64)

        self.decoder2_up1 = uppyramid_depth(256, 192,128)
        self.decoder2_up2 = uppyramid_depth(128, 192, 64)
        self.decoder2_up3 = uppyramid_depth(64, 192, 64)


        self.decoder3_up1 = uppyramid_depth(128, 192, 64)
        self.decoder3_up2 = uppyramid_depth(64, 192, 64)
        
        self.decoder4_up1 = uppyramid_depth(64, 192, 64)

        # self.decoder1_up1 = uppyramid(512*2, 64, 256)#in,swin,out
        # self.decoder1_up2 = uppyramid(256*2, 64,128)
        # self.decoder1_up3 = uppyramid(128*2, 64, 64)
        # self.decoder1_up4 = uppyramid(64*2, 64, 64)

        # self.decoder2_up1 = uppyramid_depth(256*2, 64,128)
        # self.decoder2_up2 = uppyramid_depth(128*24, 64, 64)
        # self.decoder2_up3 = uppyramid_depth(64*2, 64, 64)

        # self.decoder3_up1 = uppyramid_depth(128*2, 64, 64)
        # self.decoder3_up2 = uppyramid_depth(64*2, 64, 64)
        
        # self.decoder4_up1 = uppyramid_depth(64*2, 64, 64)
        #-----------------


    def forward(self, x, m):  ## x pre path; m, post path

        x1 = self.inc(x)  #64,224,224
        m1 = self.inc_p1(m)
        diffw1 = self.coattpost1(m1)-self.coattpre1(x1)
        m1 = m1+m1*(self.FCA1(m1-x1)+diffw1)
        d1 = m1-x1
        
        x2 = self.down1(x1) #128,112,112
        m2 = self.down_p1(m1)
        diffw2 = self.coattpost2(m2)-self.coattpre2(x2)
        m2 = m2+m2*(self.FCA2(m2-x2)+diffw2)
        d2 = m2-x2

        x3 = self.down2(x2) #256,56,56
        m3 = self.down_p2(m2)
        diffw3 = self.coattpost3(m3)-self.coattpre3(x3)
        m3 = m3+m3*(self.FCA3(m3-x3)+diffw3)
        d3 = m3-x3

        x4 = self.down3(x3) #512,28,28
        m4 = self.down_p3(m3)
        diffw4 = self.coattpost4(m4)-self.coattpre4(x4)
        m4 = m4+m4*(self.FCA4(m4-x4)+diffw4)
        d4 = m4-x4  #512,28,28

        x5 = self.down4(d4) #512,14,14
        d5 = x5 # 512, 14, 14
        
        # path diff
        # s1_out=self.swin1(m-x) # 64,112,112
        s13=self.swin13(d1)#-----64, 224,224---192,56,56
        s14=self.swin14(d1)#-----64, 224,224---192,28,28
        s15=self.swin15(d1)# ----64, 224,224---192,14,14
        s1_out=self.swin1(d1) # 64,224,224----192,112,112 
        s1 = self.swconv1(torch.cat((d2,s1_out), dim=1)) #320,112,112  #128,112,112

        s24 =self.swin24(s1_out)#---------192,112,112 -----192,28,28
        s25=self.swin25(s1_out)#----------192,112,112 ---192,14,14
        s2_out=self.swin2(s1) # ----------128,112,112- ----192,56,56
        s2 = self.swconv2(torch.cat((d3,s2_out,s13), dim=1))#640,56,56  128,56,56

        s35=self.swin35(s2_out)#-----------192,56,56  192,14,14
        s3_out=self.swin3(s2) #------------128,56,56  192,28,28
        s3 = self.swconv3(torch.cat((d4,s3_out,s14,s24), dim=1))#512+64+64+64**32,28,28 #1088,28,28  128,28,28

        s4_out=self.swin4(s3) #128,28,28  192, 14,14
        s4 = self.swconv4(torch.cat((s4_out,s15,s25,s35), dim=1))#64+64+64+64  #768,14,14  128,14,14
        # s4_out=s4
        
        x = self.decoder1_up1(d5, d4, s4)    #512+512+64       512+512+128---1152,14,14
        decoder1_4 = x #256,28,28
        x = self.decoder1_up2(x, d3, s3_out) #256,256,64       256+256+192 ---704,28,28
        decoder1_3 = x #128,56,56
        x = self.decoder1_up3(x, d2, s2_out) #128,128,64       128+128+192---448,56,56
        decoder1_2 = x # 64,112,112
        x = self.decoder1_up4(x, d1, s1_out) #64.64.64         64+64+192---320,112,112
        decoder1_1 = x # 64,224,224

        x = self.decoder2_up1(decoder1_4, decoder1_3, s3_out)#256+128+192---576,28,28
        decoder2_3 = x #128,56,56
        x = self.decoder2_up2(x, decoder1_2, s2_out)         #128+64+192---384,56,56
        decoder2_2 = x #64,112,112
        x = self.decoder2_up3(x, decoder1_1, s1_out)         #64,64,192---320,112,112
        decoder2_1 = x #64,224,224

        x = self.decoder3_up1(decoder2_3, decoder2_2, s2_out) #128+64+192---384,56,56
        decoder3_2 = x   #64,112,112
        x = self.decoder3_up2(x, decoder2_1, s1_out)          #64+64+192----384,112,112
        decoder3_1 = x   #64,224,224

        x = self.decoder4_up1(decoder3_2, decoder3_1, s1_out)# #64+64+192--------320,224,224---64,224,224

        x = self.outc(x) # 1,224,224
        # nn.Sigmoid()        
        x = self.sigmoid(x)##
        return x

#