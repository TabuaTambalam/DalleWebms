import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
import jitbase

torch.set_grad_enabled(False)

class diffusion_out(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_blocks_6_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1920,eps=0.000010)

        self.output_blocks_6_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1920, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))

        self.output_blocks_6_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=640)
        self.output_blocks_6_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.output_blocks_6_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_6_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1920, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_6_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=640, num_groups=32)
        self.output_blocks_6_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_6_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.output_blocks_6_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=640, out_features=640)
        self.output_blocks_6_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=640, out_features=640)
        self.output_blocks_6_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=640, out_features=640)
        self.output_blocks_6_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.output_blocks_6_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.output_blocks_6_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=640, out_features=640)
        self.output_blocks_6_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=640)
        self.output_blocks_6_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=640)
        self.output_blocks_6_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.output_blocks_6_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.output_blocks_6_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=640, out_features=5120)
        self.output_blocks_6_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=2560, out_features=640)
        self.output_blocks_6_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_7_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.output_blocks_7_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_7_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=640)
        self.output_blocks_7_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.output_blocks_7_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_7_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_7_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=640, num_groups=32)
        self.output_blocks_7_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_7_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.output_blocks_7_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=640, out_features=640)
        self.output_blocks_7_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=640, out_features=640)
        self.output_blocks_7_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=640, out_features=640)
        self.output_blocks_7_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.output_blocks_7_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.output_blocks_7_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=640, out_features=640)
        self.output_blocks_7_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=640)
        self.output_blocks_7_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=640)
        self.output_blocks_7_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.output_blocks_7_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.output_blocks_7_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=640, out_features=5120)
        self.output_blocks_7_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=2560, out_features=640)
        self.output_blocks_7_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_8_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=960,eps=0.000010)

        self.output_blocks_8_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=960, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_8_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=640)
        self.output_blocks_8_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.output_blocks_8_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_8_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=960, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_8_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=640, num_groups=32)
        self.output_blocks_8_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_8_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.output_blocks_8_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=640, out_features=640)
        self.output_blocks_8_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=640, out_features=640)
        self.output_blocks_8_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=640, out_features=640)
        self.output_blocks_8_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.output_blocks_8_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.output_blocks_8_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=640, out_features=640)
        self.output_blocks_8_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=640)
        self.output_blocks_8_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=640)
        self.output_blocks_8_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.output_blocks_8_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.output_blocks_8_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=640, out_features=5120)
        self.output_blocks_8_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=2560, out_features=640)
        self.output_blocks_8_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_8_2_conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_9_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=960,eps=0.000010)

        self.output_blocks_9_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=960, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_9_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=320)
        self.output_blocks_9_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.output_blocks_9_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_9_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=960, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_9_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=320, num_groups=32)
        self.output_blocks_9_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_9_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.output_blocks_9_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=320, out_features=320)
        self.output_blocks_9_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=320, out_features=320)
        self.output_blocks_9_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=320, out_features=320)
        self.output_blocks_9_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=320, out_features=320)
        self.output_blocks_9_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.output_blocks_9_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=320, out_features=320)
        self.output_blocks_9_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=320)
        self.output_blocks_9_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=320)
        self.output_blocks_9_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=320, out_features=320)
        self.output_blocks_9_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.output_blocks_9_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=320, out_features=2560)
        self.output_blocks_9_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=1280, out_features=320)
        self.output_blocks_9_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_10_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.output_blocks_10_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_10_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=320)
        self.output_blocks_10_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.output_blocks_10_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_10_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_10_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=320, num_groups=32)
        self.output_blocks_10_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_10_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.output_blocks_10_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=320, out_features=320)
        self.output_blocks_10_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=320, out_features=320)
        self.output_blocks_10_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=320, out_features=320)
        self.output_blocks_10_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=320, out_features=320)
        self.output_blocks_10_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.output_blocks_10_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=320, out_features=320)
        self.output_blocks_10_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=320)
        self.output_blocks_10_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=320)
        self.output_blocks_10_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=320, out_features=320)
        self.output_blocks_10_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.output_blocks_10_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=320, out_features=2560)
        self.output_blocks_10_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=1280, out_features=320)
        self.output_blocks_10_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_11_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.output_blocks_11_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_11_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=320)
        self.output_blocks_11_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.output_blocks_11_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_11_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_11_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=320, num_groups=32)
        self.output_blocks_11_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_11_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.output_blocks_11_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=320, out_features=320)
        self.output_blocks_11_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=320, out_features=320)
        self.output_blocks_11_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=320, out_features=320)
        self.output_blocks_11_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=320, out_features=320)
        self.output_blocks_11_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.output_blocks_11_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=320, out_features=320)
        self.output_blocks_11_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=320)
        self.output_blocks_11_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=320)
        self.output_blocks_11_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=320, out_features=320)
        self.output_blocks_11_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.output_blocks_11_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=320, out_features=2560)
        self.output_blocks_11_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=1280, out_features=320)
        self.output_blocks_11_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.out_0 = nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.out_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=4, padding=(1,1), padding_mode='zeros', stride=(1,1))


    def forward(self, v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8):
        k70 = v_0.size(0)
        k73 = v_0.size(2)
        k74 = v_0.size(3)
        k70x8=k70<<3

        v_11 = torch.cat((v_0, v_8), dim=1)
        v_15 = F.silu(v_1)
#===
        v_26 = self.output_blocks_6_0_skip_connection(v_11) + self.output_blocks_6_0_out_layers_3(F.silu(self.output_blocks_6_0_out_layers_0(self.output_blocks_6_0_in_layers_2(F.silu(self.output_blocks_6_0_in_layers_0(v_11))) + self.output_blocks_6_0_emb_layers_1(v_15).unsqueeze(2).unsqueeze(3))))

        v_30 = self.output_blocks_6_1_proj_in(self.output_blocks_6_1_norm(v_26)).permute(0,2,3,1).reshape(k70, -1, 640)
        v_31 = self.output_blocks_6_1_transformer_blocks_0_norm1(v_30)

        v_41 = self.output_blocks_6_1_transformer_blocks_0_attn1_to_q(v_31).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_39 = self.output_blocks_6_1_transformer_blocks_0_attn1_to_k(v_31).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_46 = self.output_blocks_6_1_transformer_blocks_0_attn1_to_v(v_31).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_50 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_41, v_39) * 1.118034e-01), dim=-1), v_46).reshape(k70, 8, -1, 80).permute(0,2,1,3).reshape(k70, -1, 640)
        v_52 = (self.output_blocks_6_1_transformer_blocks_0_attn1_to_out_0(v_50) + v_30)

        v_63 = self.output_blocks_6_1_transformer_blocks_0_attn2_to_q(self.output_blocks_6_1_transformer_blocks_0_norm2(v_52)).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_61 = self.output_blocks_6_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_68 = self.output_blocks_6_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_72 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_63, v_61) * 1.118034e-01), dim=-1), v_68).reshape(k70, 8, -1, 80).permute(0,2,1,3).reshape(k70, -1, 640)
        v_74 = (self.output_blocks_6_1_transformer_blocks_0_attn2_to_out_0(v_72) + v_52)

        v_77, v_78 = torch.chunk(input=self.output_blocks_6_1_transformer_blocks_0_ff_net_0_proj(self.output_blocks_6_1_transformer_blocks_0_norm3(v_74)), chunks=2, dim=-1)
        v_89 = torch.cat((v_26+self.output_blocks_6_1_proj_out((self.output_blocks_6_1_transformer_blocks_0_ff_net_2(v_77 * F.gelu(input=v_78, approximate='none')) + v_74).reshape(k70, k73, k74, 640).permute(0,3,1,2)), v_7), dim=1)
#===
        v_103 = self.output_blocks_7_0_skip_connection(v_89) + self.output_blocks_7_0_out_layers_3(F.silu(self.output_blocks_7_0_out_layers_0(self.output_blocks_7_0_in_layers_2(F.silu(self.output_blocks_7_0_in_layers_0(v_89))) + self.output_blocks_7_0_emb_layers_1(v_15).unsqueeze(2).unsqueeze(3))))

        v_107 = self.output_blocks_7_1_proj_in(self.output_blocks_7_1_norm(v_103)).permute(0,2,3,1).reshape(k70, -1, 640)
        v_108 = self.output_blocks_7_1_transformer_blocks_0_norm1(v_107)

        v_118 = self.output_blocks_7_1_transformer_blocks_0_attn1_to_q(v_108).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_116 = self.output_blocks_7_1_transformer_blocks_0_attn1_to_k(v_108).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_123 = self.output_blocks_7_1_transformer_blocks_0_attn1_to_v(v_108).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_127 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_118, v_116) * 1.118034e-01), dim=-1), v_123).reshape(k70, 8, -1, 80).permute(0,2,1,3).reshape(k70, -1, 640)
        v_129 = (self.output_blocks_7_1_transformer_blocks_0_attn1_to_out_0(v_127) + v_107)

        v_140 = self.output_blocks_7_1_transformer_blocks_0_attn2_to_q(self.output_blocks_7_1_transformer_blocks_0_norm2(v_129)).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_138 = self.output_blocks_7_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_145 = self.output_blocks_7_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_149 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_140, v_138) * 1.118034e-01), dim=-1), v_145).reshape(k70, 8, -1, 80).permute(0,2,1,3).reshape(k70, -1, 640)
        v_151 = (self.output_blocks_7_1_transformer_blocks_0_attn2_to_out_0(v_149) + v_129)

        v_154, v_155 = torch.chunk(input=self.output_blocks_7_1_transformer_blocks_0_ff_net_0_proj(self.output_blocks_7_1_transformer_blocks_0_norm3(v_151)), chunks=2, dim=-1)
        v_166 = torch.cat((v_103+ self.output_blocks_7_1_proj_out((self.output_blocks_7_1_transformer_blocks_0_ff_net_2(v_154 * F.gelu(input=v_155, approximate='none')) + v_151).reshape(k70, k73, k74, 640).permute(0,3,1,2)), v_6), dim=1)
#===
        v_180 = self.output_blocks_8_0_skip_connection(v_166) + self.output_blocks_8_0_out_layers_3(F.silu(self.output_blocks_8_0_out_layers_0(self.output_blocks_8_0_in_layers_2(F.silu(self.output_blocks_8_0_in_layers_0(v_166))) + self.output_blocks_8_0_emb_layers_1(v_15).unsqueeze(2).unsqueeze(3))))

        v_184 = self.output_blocks_8_1_proj_in(self.output_blocks_8_1_norm(v_180)).permute(0,2,3,1).reshape(k70, -1, 640)
        v_185 = self.output_blocks_8_1_transformer_blocks_0_norm1(v_184)

        v_195 = self.output_blocks_8_1_transformer_blocks_0_attn1_to_q(v_185).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_193 = self.output_blocks_8_1_transformer_blocks_0_attn1_to_k(v_185).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_200 = self.output_blocks_8_1_transformer_blocks_0_attn1_to_v(v_185).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_204 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_195, v_193) * 1.118034e-01), dim=-1), v_200).reshape(k70, 8, -1, 80).permute(0,2,1,3).reshape(k70, -1, 640)
        v_206 = (self.output_blocks_8_1_transformer_blocks_0_attn1_to_out_0(v_204) + v_184)

        v_217 = self.output_blocks_8_1_transformer_blocks_0_attn2_to_q(self.output_blocks_8_1_transformer_blocks_0_norm2(v_206)).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_215 = self.output_blocks_8_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_222 = self.output_blocks_8_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_226 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_217, v_215) * 1.118034e-01), dim=-1), v_222).reshape(k70, 8, -1, 80).permute(0,2,1,3).reshape(k70, -1, 640)
        v_228 = (self.output_blocks_8_1_transformer_blocks_0_attn2_to_out_0(v_226) + v_206)

        v_231, v_232 = torch.chunk(input=self.output_blocks_8_1_transformer_blocks_0_ff_net_0_proj(self.output_blocks_8_1_transformer_blocks_0_norm3(v_228)), chunks=2, dim=-1)
        v_240 = v_180+ self.output_blocks_8_1_proj_out( (self.output_blocks_8_1_transformer_blocks_0_ff_net_2(v_231 * F.gelu(input=v_232, approximate='none')) + v_228).reshape(k70, k73, k74, 640).permute(0,3,1,2))
        v_245 = torch.cat((self.output_blocks_8_2_conv(F.interpolate(input=v_240 , scale_factor=2.0, mode='nearest')), v_5), dim=1)

#===
        v_259 = self.output_blocks_9_0_skip_connection(v_245) + self.output_blocks_9_0_out_layers_3(F.silu(self.output_blocks_9_0_out_layers_0(self.output_blocks_9_0_in_layers_2(F.silu(self.output_blocks_9_0_in_layers_0(v_245))) + self.output_blocks_9_0_emb_layers_1(v_15).unsqueeze(2).unsqueeze(3))))

        v_263 =  self.output_blocks_9_1_proj_in(self.output_blocks_9_1_norm(v_259)).permute(0,2,3,1).reshape(k70, -1, 320)
        v_264 = self.output_blocks_9_1_transformer_blocks_0_norm1(v_263)

        v_274 = self.output_blocks_9_1_transformer_blocks_0_attn1_to_q(v_264).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_272 = self.output_blocks_9_1_transformer_blocks_0_attn1_to_k(v_264).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_279 = self.output_blocks_9_1_transformer_blocks_0_attn1_to_v(v_264).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_283 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_274, v_272) * 1.581139e-01), dim=-1), v_279).reshape(k70, 8, -1, 40).permute(0,2,1,3).reshape(k70, -1, 320)
        v_285 = (self.output_blocks_9_1_transformer_blocks_0_attn1_to_out_0(v_283) + v_263)

        v_296 = self.output_blocks_9_1_transformer_blocks_0_attn2_to_q(self.output_blocks_9_1_transformer_blocks_0_norm2(v_285)).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_294 = self.output_blocks_9_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_301 = self.output_blocks_9_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_305 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_296, v_294) * 1.581139e-01), dim=-1), v_301).reshape(k70, 8, -1, 40).permute(0,2,1,3).reshape(k70, -1, 320)
        v_307 = (self.output_blocks_9_1_transformer_blocks_0_attn2_to_out_0(v_305) + v_285)

        k73b=k73<<1
        k74b=k74<<1
        v_310, v_311 = torch.chunk(input=self.output_blocks_9_1_transformer_blocks_0_ff_net_0_proj(self.output_blocks_9_1_transformer_blocks_0_norm3(v_307)), chunks=2, dim=-1)
        v_322 = torch.cat((v_259+ self.output_blocks_9_1_proj_out((self.output_blocks_9_1_transformer_blocks_0_ff_net_2(v_310 * F.gelu(input=v_311, approximate='none')) + v_307).reshape(k70, k73b, k74b, 320).permute(0,3,1,2)), v_4), dim=1)
#===
        v_336 = self.output_blocks_10_0_skip_connection(v_322) + self.output_blocks_10_0_out_layers_3(F.silu(self.output_blocks_10_0_out_layers_0(self.output_blocks_10_0_in_layers_2(F.silu(self.output_blocks_10_0_in_layers_0(v_322))) + self.output_blocks_10_0_emb_layers_1(v_15).unsqueeze(2).unsqueeze(3))))

        v_340 = self.output_blocks_10_1_proj_in(self.output_blocks_10_1_norm(v_336)).permute(0,2,3,1).reshape(k70, -1, 320)
        v_341 = self.output_blocks_10_1_transformer_blocks_0_norm1(v_340)

        v_351 = self.output_blocks_10_1_transformer_blocks_0_attn1_to_q(v_341).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_349 = self.output_blocks_10_1_transformer_blocks_0_attn1_to_k(v_341).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_356 = self.output_blocks_10_1_transformer_blocks_0_attn1_to_v(v_341).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_360 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_351, v_349) * 1.581139e-01), dim=-1), v_356).reshape(k70, 8, -1, 40).permute(0,2,1,3).reshape(k70, -1, 320)
        v_362 = (self.output_blocks_10_1_transformer_blocks_0_attn1_to_out_0(v_360) + v_340)

        v_373 = self.output_blocks_10_1_transformer_blocks_0_attn2_to_q(self.output_blocks_10_1_transformer_blocks_0_norm2(v_362)).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_371 = self.output_blocks_10_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_378 = self.output_blocks_10_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_382 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_373, v_371) * 1.581139e-01), dim=-1), v_378).reshape(k70, 8, -1, 40).permute(0,2,1,3).reshape(k70, -1, 320)
        v_384 = (self.output_blocks_10_1_transformer_blocks_0_attn2_to_out_0(v_382) + v_362)

        v_387, v_388 = torch.chunk(input=self.output_blocks_10_1_transformer_blocks_0_ff_net_0_proj(self.output_blocks_10_1_transformer_blocks_0_norm3(v_384)), chunks=2, dim=-1)     
        v_399 = torch.cat((v_336+ self.output_blocks_10_1_proj_out((self.output_blocks_10_1_transformer_blocks_0_ff_net_2(v_387 * F.gelu(input=v_388, approximate='none')) + v_384).reshape(k70, k73b, k74b, 320).permute(0,3,1,2)), v_3), dim=1)
#===
        v_413 = self.output_blocks_11_0_skip_connection(v_399) + self.output_blocks_11_0_out_layers_3(F.silu(self.output_blocks_11_0_out_layers_0(self.output_blocks_11_0_in_layers_2(F.silu(self.output_blocks_11_0_in_layers_0(v_399))) + self.output_blocks_11_0_emb_layers_1(v_15).unsqueeze(2).unsqueeze(3))))

        v_417 = self.output_blocks_11_1_proj_in(self.output_blocks_11_1_norm(v_413)).permute(0,2,3,1).reshape(k70, -1, 320)
        v_418 = self.output_blocks_11_1_transformer_blocks_0_norm1(v_417)

        v_428 = self.output_blocks_11_1_transformer_blocks_0_attn1_to_q(v_418).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_426 = self.output_blocks_11_1_transformer_blocks_0_attn1_to_k(v_418).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_433 = self.output_blocks_11_1_transformer_blocks_0_attn1_to_v(v_418).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_437 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_428, v_426) * 1.581139e-01), dim=-1), v_433).reshape(k70, 8, -1, 40).permute(0,2,1,3).reshape(k70, -1, 320)
        v_439 = (self.output_blocks_11_1_transformer_blocks_0_attn1_to_out_0(v_437) + v_417)

        v_444 = self.output_blocks_11_1_transformer_blocks_0_attn2_to_q(self.output_blocks_11_1_transformer_blocks_0_norm2(v_439)).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_445 = self.output_blocks_11_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_446 = self.output_blocks_11_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_459 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=torch.einsum('i j l, i k l -> i j k', v_444, v_445)* 1.581139e-01, dim=-1), v_446).reshape(k70, 8, -1, 40).permute(0,2,1,3).reshape(k70, -1, 320)
        v_461 =(self.output_blocks_11_1_transformer_blocks_0_attn2_to_out_0(v_459) + v_439)

        v_464, v_465 = torch.chunk(input=self.output_blocks_11_1_transformer_blocks_0_ff_net_0_proj(self.output_blocks_11_1_transformer_blocks_0_norm3(v_461)), chunks=2, dim=-1)
        v_475 = v_413+self.output_blocks_11_1_proj_out((self.output_blocks_11_1_transformer_blocks_0_ff_net_2(v_464 * F.gelu(input=v_465, approximate='none')) + v_461).reshape(k70, k73b, k74b, 320).permute(0,3,1,2))

        return self.out_2(F.silu(self.out_0(v_475)))

with init_empty_weights():
    jitbase.diffusion_out_base= diffusion_out().requires_grad_(False).eval()