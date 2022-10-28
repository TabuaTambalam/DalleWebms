import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.post_quant_conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=3, kernel_size=(1,1), out_channels=3, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.decoder_conv_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=3, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_mid_block_1_norm1 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=512, num_groups=32)
        self.decoder_mid_block_1_conv1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_mid_block_1_norm2 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=512, num_groups=32)
        self.decoder_mid_block_1_conv2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_mid_attn_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=512, num_groups=32)
        self.decoder_mid_attn_1_q = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(1,1), out_channels=512, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.decoder_mid_attn_1_k = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(1,1), out_channels=512, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.decoder_mid_attn_1_v = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(1,1), out_channels=512, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.decoder_mid_attn_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(1,1), out_channels=512, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.decoder_mid_block_2_norm1 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=512, num_groups=32)
        self.decoder_mid_block_2_conv1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_mid_block_2_norm2 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=512, num_groups=32)
        self.decoder_mid_block_2_conv2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_2_block_0_norm1 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=512, num_groups=32)
        self.decoder_up_2_block_0_conv1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_2_block_0_norm2 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=512, num_groups=32)
        self.decoder_up_2_block_0_conv2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_2_block_1_norm1 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=512, num_groups=32)
        self.decoder_up_2_block_1_conv1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_2_block_1_norm2 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=512, num_groups=32)
        self.decoder_up_2_block_1_conv2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_2_block_2_norm1 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=512, num_groups=32)
        self.decoder_up_2_block_2_conv1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_2_block_2_norm2 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=512, num_groups=32)
        self.decoder_up_2_block_2_conv2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_2_upsample_conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_1_block_0_norm1 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=512, num_groups=32)
        self.decoder_up_1_block_0_conv1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_1_block_0_norm2 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=256, num_groups=32)
        self.decoder_up_1_block_0_conv2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_1_block_0_nin_shortcut = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(1,1), out_channels=256, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.decoder_up_1_block_1_norm1 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=256, num_groups=32)
        self.decoder_up_1_block_1_conv1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_1_block_1_norm2 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=256, num_groups=32)
        self.decoder_up_1_block_1_conv2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_1_block_2_norm1 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=256, num_groups=32)
        self.decoder_up_1_block_2_conv1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_1_block_2_norm2 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=256, num_groups=32)
        self.decoder_up_1_block_2_conv2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_1_upsample_conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_0_block_0_norm1 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=256, num_groups=32)
        self.decoder_up_0_block_0_conv1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_0_block_0_norm2 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=128, num_groups=32)
        self.decoder_up_0_block_0_conv2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_0_block_0_nin_shortcut = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.decoder_up_0_block_1_norm1 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=128, num_groups=32)
        self.decoder_up_0_block_1_conv1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_0_block_1_norm2 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=128, num_groups=32)
        self.decoder_up_0_block_1_conv2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_0_block_2_norm1 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=128, num_groups=32)
        self.decoder_up_0_block_2_conv1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_up_0_block_2_norm2 = nn.GroupNorm(affine=True, eps=0.000001, num_channels=128, num_groups=32)
        self.decoder_up_0_block_2_conv2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.decoder_norm_out = nn.GroupNorm(affine=True, eps=0.000001, num_channels=128, num_groups=32)
        self.decoder_conv_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=3, padding=(1,1), padding_mode='zeros', stride=(1,1))


    def forward(self, v_0):
        v_2 = self.decoder_conv_in(self.post_quant_conv(v_0))

        v_9 = v_2 + self.decoder_mid_block_1_conv2(F.silu(input=self.decoder_mid_block_1_norm2(self.decoder_mid_block_1_conv1(F.silu(input=self.decoder_mid_block_1_norm1(v_2))))))
        v_10 = self.decoder_mid_attn_1_norm(v_9)
        v_16 = self.decoder_mid_attn_1_q(v_10).reshape(1, 512, -1).permute(0,2,1)
        v_15 = self.decoder_mid_attn_1_k(v_10).reshape(1, 512, -1)
        v_20 = self.decoder_mid_attn_1_v(v_10).reshape(1, 512, -1)

        v_25 = v_9 + self.decoder_mid_attn_1_proj_out(torch.bmm(input=v_20, mat2=F.softmax(input=(torch.bmm(input=v_16, mat2=v_15) * 4.419400e-02), dim=2).permute(0,2,1)).reshape(1, 512, 128, 128))

        v_32 = v_25 + self.decoder_mid_block_2_conv2(F.silu(input=self.decoder_mid_block_2_norm2(self.decoder_mid_block_2_conv1(F.silu(input=self.decoder_mid_block_2_norm1(v_25))))))

        v_39 = v_32 + self.decoder_up_2_block_0_conv2(F.silu(input=self.decoder_up_2_block_0_norm2(self.decoder_up_2_block_0_conv1(F.silu(input=self.decoder_up_2_block_0_norm1(v_32))))))

        v_46 = v_39 + self.decoder_up_2_block_1_conv2(F.silu(input=self.decoder_up_2_block_1_norm2(self.decoder_up_2_block_1_conv1(F.silu(input=self.decoder_up_2_block_1_norm1(v_39))))))

        v_53 = v_46 + self.decoder_up_2_block_2_conv2(F.silu(input=self.decoder_up_2_block_2_norm2(self.decoder_up_2_block_2_conv1(F.silu(input=self.decoder_up_2_block_2_norm1(v_46))))))
        v_55 = self.decoder_up_2_upsample_conv(F.interpolate(v_53, scale_factor=2.0,mode='nearest'))

        v_63 = self.decoder_up_1_block_0_nin_shortcut(v_55) + self.decoder_up_1_block_0_conv2(F.silu(input=self.decoder_up_1_block_0_norm2(self.decoder_up_1_block_0_conv1(F.silu(input=self.decoder_up_1_block_0_norm1(v_55))))))

        v_70 = v_63 + self.decoder_up_1_block_1_conv2(F.silu(input=self.decoder_up_1_block_1_norm2(self.decoder_up_1_block_1_conv1(F.silu(input=self.decoder_up_1_block_1_norm1(v_63))))))

        v_77 = v_70 + self.decoder_up_1_block_2_conv2(F.silu(input=self.decoder_up_1_block_2_norm2(self.decoder_up_1_block_2_conv1(F.silu(input=self.decoder_up_1_block_2_norm1(v_70))))))
        v_79 = self.decoder_up_1_upsample_conv(F.interpolate(v_77, scale_factor=2.0,mode='nearest'))

        v_87 = self.decoder_up_0_block_0_nin_shortcut(v_79) + self.decoder_up_0_block_0_conv2(F.silu(input=self.decoder_up_0_block_0_norm2(self.decoder_up_0_block_0_conv1(F.silu(input=self.decoder_up_0_block_0_norm1(v_79))))))

        v_94 = v_87 + self.decoder_up_0_block_1_conv2(F.silu(input=self.decoder_up_0_block_1_norm2(self.decoder_up_0_block_1_conv1(F.silu(input=self.decoder_up_0_block_1_norm1(v_87))))))

        v_101 = v_94 + self.decoder_up_0_block_2_conv2(F.silu(input=self.decoder_up_0_block_2_norm2(self.decoder_up_0_block_2_conv1(F.silu(input=self.decoder_up_0_block_2_norm1(v_94))))))

        return self.decoder_conv_out(F.silu(input=self.decoder_norm_out(v_101)))
