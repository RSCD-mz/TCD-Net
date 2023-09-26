import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np

##single_Swin_block

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape(B, H//window_size, window_size, W//window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5)  #[B, H//window_size, W//window_size, window_size, window_size, C]
    x = x.reshape(-1, window_size, window_size, C) #[B*num_window, window_size, window_size, C]

    return x

def window_reverse(windows, window_size, H, W):
    #windows:[B*num_window, window_size, window_size, C]
    B = int(windows.shape[0] // (H/window_size * W/window_size))
    #x: [B, H//window_size, W//window_size, window_size, window_size, C]
    x = windows.reshape(B, H//window_size, W//window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5) #[B, H//window_size, window_size, W//window_size, window_size, C]

    x = x.reshape(B, H, W, -1)
    return x

def generate_mask(input_res, window_size, shift_size):
    H, W,= input_res
    Hp = int(np.ceil(H / window_size)) * window_size
    Wp = int(np.ceil(W / window_size)) * window_size

    image_mask = torch.zeros((1, Hp, Wp, 1))
    h_slice = (slice(0,-window_size),
               slice(-window_size, -shift_size),
               slice(-shift_size, None)
    )

    w_slice = (slice(0,-window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None)
    )

    cnt = 0
    for h in h_slice:
        for w in w_slice:
            image_mask[:, h, w, :] = cnt
            cnt += 1
    mask_window = window_partition(image_mask, window_size)
    mask_window = mask_window.reshape(-1, window_size*window_size)

    attn_mask = mask_window.unsqueeze(1) - mask_window.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class Patch_Embeding(nn.Module):
    def __init__(self, chan=3, dim = 96, patch_size = 4):
        super().__init__()
        self.patch = nn.Conv2d(chan, dim, kernel_size= patch_size, stride = patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.patch(x) #[B, C, H, W] , C = dim
        x = x.flatten(2).transpose(1, 2)   #[B, num_patches, C]
        x = self.norm(x)
        return x

class Patch_Merging(nn.Module):
    def __init__(self, input_res, dim):
        super().__init__()
        self.resolution = input_res
        self.dim = dim

        self.reduction = nn.Linear(4*dim, 2*dim)
        self.norm = nn.LayerNorm(2*dim)

    def forward(self, x):
        # x: [B, num_patches, C]
        H, W = self.resolution
        B, _, C = x.shape

        x = x.reshape(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat((x0, x1, x2, x3), -1)

        x = x.reshape(B, -1, 4*C)
        x = self.reduction(x)
        x = self.norm(x)

        return x

#Swin_block


class window_attention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False):
        super().__init__()

        self.num_heads = num_heads
        prehead_dim = dim // self.num_heads
        self.scale = prehead_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask = None):
        # x: [B*num_window, num_patches, embed_dim]
        B, num_patches, total_dim = x.shape

        qkv = self.qkv(x) #[B*num_window,, num_patches, 3*embed_dim]

        qkv = qkv.reshape(B, num_patches, 3, self.num_heads, total_dim // self.num_heads) #[B*num_window,, num_patches, 3, num_heads, prehead_dim]

        qkv = qkv.permute(2, 0, 3, 1, 4) #[3, B*num_window,, num_heads, num_patches, prehead_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  #[B*num_window,, num_heads, num_patches, prehead_dim]

        atten = (q @ k.transpose(-2,-1)) * self.scale  #[B*num_window,, num_heads, num_patches, num_patches]
        if mask is None:
            atten = atten.softmax(dim=-1)
        else:
            #mask: [num_window, num_patches, num_patches]
            #atten: [B*num_window, num_head, num_patches, num_patches]
            atten = atten.reshape(B//mask.shape[0], mask.shape[0], self.num_heads, mask.shape[1], mask.shape[1])
            #reshape_atten [B, num_window, num_head, num_patches, num_patches]
            #mask [1, num_window, 1, num_patches, num_patches]
            atten = atten + mask.unsqueeze(1).unsqueeze(0).cuda()# atten = atten + mask.unsqueeze(1).unsqueeze(0)
            atten = atten.reshape(-1, self.num_heads, mask.shape[1], mask.shape[1]) #[B*num_window, num_head, num_patches, num_patches]
            atten = atten.softmax(dim = -1)

        atten = atten @ v  ## [B, num_heads, num_patches, prehead_dim]
        atten = atten.transpose(1,2) #[B, num_patches+1, num_heads, prehead_dim]
        atten = atten.reshape(B, num_patches, total_dim)  #[B, num_patches+1, embed_dim]

        out = self.proj(atten)

        return out

class MLP(nn.Module):
    def __init__(self, in_dim, mlp_ratio = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim*mlp_ratio)
        self.actlayer = nn.GELU()
        self.fc2 = nn.Linear(mlp_ratio*in_dim, in_dim)

    def forward(self, x):
        x = self.fc1(x)  #[B, num_patches+1, hidden_dim]
        x = self.actlayer(x)
        x = self.fc2(x)  #[B, num_patches+1, out_dim]
        x = self.actlayer(x)

        return x

# swin_encode & Patch_Merging
class Swin_Block(nn.Module):
    def __init__(self, dim, num_heads, Out_res, window_size, qkv_bias = False, shift_size=0):
        super().__init__()

        self.dim = dim
        self.resolution = Out_res
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size =shift_size
        self.atten_norm = nn.LayerNorm(dim)
        self.atten = window_attention(dim, num_heads, qkv_bias)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=4)

        #self.patch_merging = Patch_Merging(input_res, dim)


    def forward(self, x):
        # x:[B, num_patches, embed_dim]
        H, W = self.resolution
        B, N, C = x.shape
        assert N == H * W

        h = x
        x = self.atten_norm(x)
        x = x.reshape(B, H, W, C)

        if self.shift_size > 0:
            shift_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
            atten_mask = generate_mask(input_res=self.resolution, window_size=self.window_size, shift_size=self.shift_size)
        else:
            shift_x = x
            atten_mask = None
        

        x_window = window_partition(shift_x, self.window_size) #[B*num_patches, window_size, window_size, C]
        x_window = x_window.reshape(-1, self.window_size*self.window_size, C)
        atten_window = self.atten(x_window, mask = atten_mask) #[B*num_patches, window_size*window_size, C]
        atten_window = atten_window.reshape(-1, self.window_size, self.window_size, C)
        x = window_reverse(atten_window, self.window_size, H, W) #[B, H, W, C]
        x = x.reshape(B, -1, C)
        x = h + x

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x

        return x

class Swin_Model(nn.Module):
    def __init__(self,
                 chan,
                 dim,
                 patch_size,
                 num_heads,
                 input_res,
                 window_size,
                 qkv_bias = None
                 ):
        super().__init__()

        self.patch_embed = Patch_Embeding(chan,dim, patch_size)
        self.W_MSA_block = Swin_Block(dim, num_heads, input_res, window_size, qkv_bias, shift_size=0)
        self.SW_MSA_block = Swin_Block(dim, num_heads, input_res, window_size, qkv_bias, shift_size=window_size//2)
        self.patch_merging = Patch_Merging(input_res, dim)
    def forward(self, x):

        x = self.patch_embed(x)
        x = self.W_MSA_block(x)
        x = self.SW_MSA_block(x)
        out = self.patch_merging(x)

        h, w = int(np.sqrt(out.size(1))), int(np.sqrt(out.size(1)))
        x = out.permute(0, 2, 1)
        x = x.contiguous().view(out.size(0), out.size(2), h, w)

        return x



# def main():
#     input = torch.randn((2, 128, 112,112))

#     n,c,h,w =input.size()
#     p_size = 2
#     outc = 2*c

#     model = Swin_Model(
#         chan=c,
#         dim=outc,
#         patch_size=p_size,
#         num_heads=p_size,
#         input_res=[h//p_size, w//p_size],
#         window_size=7,
#         qkv_bias=True
#     )
#     print(model)
#     out = model(input)
#     print(out.size())  
#     fig = plt.figure(figsize=(h, w))
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
#     for i in range(5):
#         ax = fig.add_subplot(12, 5, i+1, xticks=[], yticks=[])
#         ax.imshow(x[0][i].detach().numpy(), cmap="gray")
 
#     plt.show()

#     print(out.size())  #[B, H/8 * W/8, 2*embed_dim]
    

# # def main():



# if __name__ == '__main__':
#     main()