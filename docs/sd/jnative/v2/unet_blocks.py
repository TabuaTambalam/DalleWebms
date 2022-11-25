import torch
import torch.nn as nn
import torch.nn.functional as F

from .jitbase import config

class ResnetBlock(nn.Module):
    def __init__(
        self,
        chn_in,
        chn_out,
        prv_skip=False
    ):
        super().__init__()
        self.in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=chn_in,eps=1e-5)
        self.in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=chn_in, kernel_size=(3,3), out_channels=chn_out, padding=(1,1), padding_mode=config.pad, stride=(1,1))
        self.emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=chn_out)
        self.out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=chn_out,eps=1e-5)
        self.out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=chn_out, kernel_size=(3,3), out_channels=chn_out, padding=(1,1), padding_mode=config.pad, stride=(1,1))
        self.skip = prv_skip
        if prv_skip:
            self.skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=chn_in, kernel_size=(1,1), out_channels=chn_out, padding=(0,0), padding_mode=config.pad, stride=(1,1))

    def forward(self, x, t):
        z=x
        if self.skip:
            z=self.skip_connection(x)
        return z + self.out_layers_3(F.silu(self.out_layers_0(self.in_layers_2(F.silu(self.in_layers_0(x))) + self.emb_layers_1(t).unsqueeze(2).unsqueeze(3))))

class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        features,
        cat_prev=False
    ):
        super().__init__()
        self.chn=channels
        self.cat_prev=cat_prev
        self.s_attnK=self.modify0
        self.s_attnV=self.modify0
        self.c_attnK=self.modify0
        self.c_attnV=self.modify0
        self.norm = nn.GroupNorm(affine=True, eps=1e-6, num_channels=channels, num_groups=32)
        self.proj_in = nn.Linear(bias=True, in_features=channels, out_features=channels)
        self.transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=1e-5, normalized_shape=(channels,))
        self.transformer_blocks_0_attn1 = nn.MultiheadAttention(channels, 8,batch_first=True)
        self.transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=1e-5, normalized_shape=(channels,))
        self.transformer_blocks_0_attn2 =  nn.MultiheadAttention(channels, 8,kdim=config.txt_dim,vdim=config.txt_dim,batch_first=True)
        self.transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=1e-5, normalized_shape=(channels,))
        self.transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=channels, out_features=features<<1)
        self.transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=features, out_features=channels)
        self.proj_out = nn.Linear(bias=True, in_features=channels, out_features=channels)

    def modify0(self,x):
        return x

    def forward(self, x, cond_k,cond_v):
        b=int(x.size(0))
        h=int(x.size(2))
        w=int(x.size(3))
        LastO = self.proj_in(self.norm(x).permute(0,2,3,1).reshape(b, -1, self.chn))

        v_185 = self.transformer_blocks_0_norm1(LastO)
        LastO = LastO + self.transformer_blocks_0_attn1(v_185,self.s_attnK(v_185),self.s_attnV(v_185),need_weights=False)[0]

        v_228 = LastO + self.transformer_blocks_0_attn2(self.transformer_blocks_0_norm2(LastO),self.c_attnK(cond_k),self.c_attnV(cond_v),need_weights=False)[0]

        v_231, v_232 = torch.chunk(input=self.transformer_blocks_0_ff_net_0_proj(self.transformer_blocks_0_norm3(v_228)), chunks=2, dim=-1)
        return  x+ self.proj_out((v_228+ self.transformer_blocks_0_ff_net_2(v_231 * F.gelu(input=v_232, approximate='none')))).reshape(b, h, w, self.chn).permute(0,3,1,2)


class AttentionBlock_conv(nn.Module):
    def __init__(
        self,
        chn,
        stride
    ):
        super().__init__()
        if stride == 2:
            self.op = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=chn, kernel_size=(3,3), out_channels=chn, padding=(1,1), padding_mode=config.pad, stride=(2,2))
            self.forward=self.forward_op
        elif stride == 1:
            self.conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=chn, kernel_size=(3,3), out_channels=chn, padding=(1,1), padding_mode=config.pad, stride=(1,1))
            self.forward=self.forward_conv
    def forward_op(self, x):
        return self.op(x)
    def forward_conv(self, x):
        return self.conv(F.interpolate(input=x , scale_factor=2.0, mode='nearest'))
