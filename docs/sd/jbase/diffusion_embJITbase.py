import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
import jitbase

torch.set_grad_enabled(False)

class diffusion_emb(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embed_0 = nn.Linear(bias=True, in_features=320, out_features=1280)

        self.time_embed_2 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.input_blocks_0_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=4, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_1_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.input_blocks_1_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))

        self.input_blocks_1_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=320)
        self.input_blocks_1_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.input_blocks_1_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_1_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=320, num_groups=32)
        self.input_blocks_1_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.input_blocks_1_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.input_blocks_1_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=320, out_features=320)
        self.input_blocks_1_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=320, out_features=320)
        self.input_blocks_1_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=320, out_features=320)
        self.input_blocks_1_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=320, out_features=320)
        self.input_blocks_1_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.input_blocks_1_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=320, out_features=320)
        self.input_blocks_1_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=320)
        self.input_blocks_1_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=320)
        self.input_blocks_1_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=320, out_features=320)
        self.input_blocks_1_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.input_blocks_1_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=320, out_features=2560)
        self.input_blocks_1_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=1280, out_features=320)
        self.input_blocks_1_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.input_blocks_2_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.input_blocks_2_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_2_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=320)
        self.input_blocks_2_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.input_blocks_2_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_2_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=320, num_groups=32)
        self.input_blocks_2_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.input_blocks_2_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.input_blocks_2_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=320, out_features=320)
        self.input_blocks_2_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=320, out_features=320)
        self.input_blocks_2_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=320, out_features=320)
        self.input_blocks_2_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=320, out_features=320)
        self.input_blocks_2_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.input_blocks_2_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=320, out_features=320)
        self.input_blocks_2_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=320)
        self.input_blocks_2_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=320)
        self.input_blocks_2_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=320, out_features=320)
        self.input_blocks_2_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(320,))
        self.input_blocks_2_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=320, out_features=2560)
        self.input_blocks_2_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=1280, out_features=320)
        self.input_blocks_2_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.input_blocks_3_0_op = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.input_blocks_4_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.input_blocks_4_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_4_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=640)
        self.input_blocks_4_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.input_blocks_4_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_4_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.input_blocks_4_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=640, num_groups=32)
        self.input_blocks_4_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.input_blocks_4_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.input_blocks_4_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=640, out_features=640)
        self.input_blocks_4_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=640, out_features=640)
        self.input_blocks_4_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=640, out_features=640)
        self.input_blocks_4_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.input_blocks_4_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.input_blocks_4_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=640, out_features=640)
        self.input_blocks_4_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=640)
        self.input_blocks_4_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=640)
        self.input_blocks_4_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.input_blocks_4_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.input_blocks_4_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=640, out_features=5120)
        self.input_blocks_4_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=2560, out_features=640)
        self.input_blocks_4_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.input_blocks_5_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.input_blocks_5_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_5_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=640)
        self.input_blocks_5_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.input_blocks_5_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_5_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=640, num_groups=32)
        self.input_blocks_5_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.input_blocks_5_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.input_blocks_5_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=640, out_features=640)
        self.input_blocks_5_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=640, out_features=640)
        self.input_blocks_5_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=640, out_features=640)
        self.input_blocks_5_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.input_blocks_5_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.input_blocks_5_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=640, out_features=640)
        self.input_blocks_5_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=640)
        self.input_blocks_5_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=640)
        self.input_blocks_5_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.input_blocks_5_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(640,))
        self.input_blocks_5_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=640, out_features=5120)
        self.input_blocks_5_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=2560, out_features=640)
        self.input_blocks_5_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.input_blocks_6_0_op = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.input_blocks_7_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.input_blocks_7_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_7_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.input_blocks_7_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.input_blocks_7_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_7_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.input_blocks_7_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=1280, num_groups=32)
        self.input_blocks_7_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.input_blocks_7_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.input_blocks_7_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=1280, out_features=1280)
        self.input_blocks_7_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=1280, out_features=1280)
        self.input_blocks_7_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=1280, out_features=1280)
        self.input_blocks_7_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.input_blocks_7_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.input_blocks_7_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=1280, out_features=1280)
        self.input_blocks_7_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=1280)
        self.input_blocks_7_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=1280)
        self.input_blocks_7_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.input_blocks_7_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.input_blocks_7_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=1280, out_features=10240)
        self.input_blocks_7_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=5120, out_features=1280)
        self.input_blocks_7_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.input_blocks_8_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.input_blocks_8_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_8_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.input_blocks_8_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.input_blocks_8_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_8_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=1280, num_groups=32)


        self.input_blocks_8_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.input_blocks_8_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.input_blocks_8_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=1280, out_features=1280)
        self.input_blocks_8_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=1280, out_features=1280)
        self.input_blocks_8_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=1280, out_features=1280)
        self.input_blocks_8_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.input_blocks_8_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.input_blocks_8_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=1280, out_features=1280)
        self.input_blocks_8_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=1280)
        self.input_blocks_8_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=1280)
        self.input_blocks_8_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.input_blocks_8_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.input_blocks_8_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=1280, out_features=10240)
        self.input_blocks_8_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=5120, out_features=1280)
        self.input_blocks_8_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.input_blocks_9_0_op = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.input_blocks_10_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.input_blocks_10_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_10_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.input_blocks_10_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.input_blocks_10_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_11_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.input_blocks_11_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_11_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.input_blocks_11_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.input_blocks_11_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.middle_block_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.middle_block_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.middle_block_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.middle_block_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.middle_block_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.middle_block_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=1280, num_groups=32)
        self.middle_block_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.middle_block_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.middle_block_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=1280, out_features=1280)
        self.middle_block_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=1280, out_features=1280)
        self.middle_block_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=1280, out_features=1280)
        self.middle_block_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.middle_block_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.middle_block_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=1280, out_features=1280)
        self.middle_block_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=1280)
        self.middle_block_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=1280)
        self.middle_block_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.middle_block_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.middle_block_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=1280, out_features=10240)
        self.middle_block_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=5120, out_features=1280)
        self.middle_block_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.middle_block_2_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.middle_block_2_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.middle_block_2_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.middle_block_2_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.middle_block_2_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.freqs=nn.Parameter(torch.ones(1), requires_grad=False)


    def forward(self, v_0, v_1, v_2):
        k70 = v_0.size(0)
        k73 = v_0.size(2)
        k74 = v_0.size(3)
        k70x8=k70<<3

        v_5 = (v_1.unsqueeze(1) * self.freqs)
        v_12 = self.input_blocks_0_0(v_0)
        v_11 = self.time_embed_2(F.silu(self.time_embed_0(torch.cat((torch.cos(v_5), torch.sin(v_5)), dim=-1))))
        v_18 = F.silu(v_11)

#sqrt(1/40)=1.581139e-01
#===
        v_28 = v_12+self.input_blocks_1_0_out_layers_3(F.silu(self.input_blocks_1_0_out_layers_0(self.input_blocks_1_0_in_layers_2(F.silu(self.input_blocks_1_0_in_layers_0(v_12))) + self.input_blocks_1_0_emb_layers_1(v_18).unsqueeze(2).unsqueeze(3))))

        v_32 = self.input_blocks_1_1_proj_in(self.input_blocks_1_1_norm(v_28)).permute(0,2,3,1).reshape(k70, -1, 320)
        v_33 = self.input_blocks_1_1_transformer_blocks_0_norm1(v_32)

        v_43 = self.input_blocks_1_1_transformer_blocks_0_attn1_to_q(v_33).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_41 = self.input_blocks_1_1_transformer_blocks_0_attn1_to_k(v_33).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_48 = self.input_blocks_1_1_transformer_blocks_0_attn1_to_v(v_33).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_52 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_43, v_41) * 1.581139e-01), dim=-1), v_48).reshape(k70, 8, -1, 40).permute(0,2,1,3).reshape(k70, -1, 320)
        v_54 = (self.input_blocks_1_1_transformer_blocks_0_attn1_to_out_0(v_52) + v_32)

        v_65 = self.input_blocks_1_1_transformer_blocks_0_attn2_to_q(self.input_blocks_1_1_transformer_blocks_0_norm2(v_54)).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_63 = self.input_blocks_1_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_70 = self.input_blocks_1_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_74 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_65, v_63) * 1.581139e-01), dim=-1), v_70).reshape(k70, 8, -1, 40).permute(0,2,1,3).reshape(k70, -1, 320)
        v_76 = (self.input_blocks_1_1_transformer_blocks_0_attn2_to_out_0(v_74) + v_54)

        v_79, v_80 = torch.chunk(input=self.input_blocks_1_1_transformer_blocks_0_ff_net_0_proj(self.input_blocks_1_1_transformer_blocks_0_norm3(v_76)), chunks=2, dim=-1)
        v_90 = v_28 + self.input_blocks_1_1_proj_out((self.input_blocks_1_1_transformer_blocks_0_ff_net_2(v_79 * F.gelu(input=v_80, approximate='none')) + v_76).reshape(k70, k73, k74, 320).permute(0,3,1,2))
#===
        v_103 = v_90 + self.input_blocks_2_0_out_layers_3(F.silu(self.input_blocks_2_0_out_layers_0(self.input_blocks_2_0_in_layers_2(F.silu(self.input_blocks_2_0_in_layers_0(v_90))) + self.input_blocks_2_0_emb_layers_1(v_18).unsqueeze(2).unsqueeze(3))))

        v_107 = self.input_blocks_2_1_proj_in(self.input_blocks_2_1_norm(v_103)).permute(0,2,3,1).reshape(k70, -1, 320)
        v_108 = self.input_blocks_2_1_transformer_blocks_0_norm1(v_107)

        v_118 = self.input_blocks_2_1_transformer_blocks_0_attn1_to_q(v_108).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_116 = self.input_blocks_2_1_transformer_blocks_0_attn1_to_k(v_108).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_123 = self.input_blocks_2_1_transformer_blocks_0_attn1_to_v(v_108).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_127 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_118, v_116) * 1.581139e-01), dim=-1), v_123).reshape(k70, 8, -1, 40).permute(0,2,1,3).reshape(k70, -1, 320)
        v_129 = (self.input_blocks_2_1_transformer_blocks_0_attn1_to_out_0(v_127) + v_107)

        v_140 = self.input_blocks_2_1_transformer_blocks_0_attn2_to_q(self.input_blocks_2_1_transformer_blocks_0_norm2(v_129)).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_138 = self.input_blocks_2_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_145 =self.input_blocks_2_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, -1, 8, 40).permute(0,2,1,3).reshape(k70x8, -1, 40)
        v_149 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_140, v_138) * 1.581139e-01), dim=-1), v_145).reshape(k70, 8, -1, 40).permute(0,2,1,3).reshape(k70, -1, 320)
        v_151 = (self.input_blocks_2_1_transformer_blocks_0_attn2_to_out_0(v_149) + v_129)

        v_154, v_155 = torch.chunk(input=self.input_blocks_2_1_transformer_blocks_0_ff_net_0_proj(self.input_blocks_2_1_transformer_blocks_0_norm3(v_151)), chunks=2, dim=-1)
        v_163 = v_103+ self.input_blocks_2_1_proj_out((self.input_blocks_2_1_transformer_blocks_0_ff_net_2(v_154 * F.gelu(input=v_155, approximate='none')) + v_151).reshape(k70, k73, k74, 320).permute(0,3,1,2))
#===
        v_164 = self.input_blocks_3_0_op(v_163)

        v_180 = self.input_blocks_4_0_skip_connection(v_164) + self.input_blocks_4_0_out_layers_3(F.silu(self.input_blocks_4_0_out_layers_0(self.input_blocks_4_0_in_layers_2(F.silu(self.input_blocks_4_0_in_layers_0(v_164))) + self.input_blocks_4_0_emb_layers_1(v_18).unsqueeze(2).unsqueeze(3))))

        v_184 = self.input_blocks_4_1_proj_in(self.input_blocks_4_1_norm(v_180)).permute(0,2,3,1).reshape(k70, -1, 640)
        v_185 = self.input_blocks_4_1_transformer_blocks_0_norm1(v_184)

        v_195 = self.input_blocks_4_1_transformer_blocks_0_attn1_to_q(v_185).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_193 = self.input_blocks_4_1_transformer_blocks_0_attn1_to_k(v_185).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_200 = self.input_blocks_4_1_transformer_blocks_0_attn1_to_v(v_185).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_204 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_195, v_193) * 1.118034e-01), dim=-1), v_200).reshape(k70, 8, -1, 80).permute(0,2,1,3).reshape(k70, -1, 640)
        v_206 = (self.input_blocks_4_1_transformer_blocks_0_attn1_to_out_0(v_204) + v_184)

        v_217 = self.input_blocks_4_1_transformer_blocks_0_attn2_to_q(self.input_blocks_4_1_transformer_blocks_0_norm2(v_206)).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_215 = self.input_blocks_4_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_222 = self.input_blocks_4_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)        
        v_226 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_217, v_215) * 1.118034e-01), dim=-1), v_222).reshape(k70, 8, -1, 80).permute(0,2,1,3).reshape(k70, -1, 640)
        v_228 = (self.input_blocks_4_1_transformer_blocks_0_attn2_to_out_0(v_226) + v_206)

        k73a=k73>>1
        k74a=k74>>1
        v_231, v_232 = torch.chunk(input=self.input_blocks_4_1_transformer_blocks_0_ff_net_0_proj(self.input_blocks_4_1_transformer_blocks_0_norm3(v_228)), chunks=2, dim=-1)
        v_242 = v_180+ self.input_blocks_4_1_proj_out((v_228+ self.input_blocks_4_1_transformer_blocks_0_ff_net_2(v_231 * F.gelu(input=v_232, approximate='none'))).reshape(k70, k73a, k74a, 640).permute(0,3,1,2))
#===
        v_255 = v_242 + self.input_blocks_5_0_out_layers_3(F.silu(self.input_blocks_5_0_out_layers_0(self.input_blocks_5_0_in_layers_2(F.silu(self.input_blocks_5_0_in_layers_0(v_242))) + self.input_blocks_5_0_emb_layers_1(v_18).unsqueeze(2).unsqueeze(3))))

        v_259 = self.input_blocks_5_1_proj_in(self.input_blocks_5_1_norm(v_255)).permute(0,2,3,1).reshape(k70, -1, 640)
        v_260 = self.input_blocks_5_1_transformer_blocks_0_norm1(v_259)

        v_270 = self.input_blocks_5_1_transformer_blocks_0_attn1_to_q(v_260).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_268 = self.input_blocks_5_1_transformer_blocks_0_attn1_to_k(v_260).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_275 = self.input_blocks_5_1_transformer_blocks_0_attn1_to_v(v_260).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_279 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_270, v_268) * 1.118034e-01), dim=-1), v_275).reshape(k70, 8, -1, 80).permute(0,2,1,3).reshape(k70, -1, 640)
        v_281 = (self.input_blocks_5_1_transformer_blocks_0_attn1_to_out_0(v_279) + v_259)

        v_292 = self.input_blocks_5_1_transformer_blocks_0_attn2_to_q(self.input_blocks_5_1_transformer_blocks_0_norm2(v_281)).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_290 = self.input_blocks_5_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_297 = self.input_blocks_5_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, -1, 8, 80).permute(0,2,1,3).reshape(k70x8, -1, 80)
        v_301 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_292, v_290) * 1.118034e-01), dim=-1), v_297).reshape(k70, 8, -1, 80).permute(0,2,1,3).reshape(k70, -1, 640)
        v_303 = (self.input_blocks_5_1_transformer_blocks_0_attn2_to_out_0(v_301) + v_281)


        v_306, v_307 = torch.chunk(input=self.input_blocks_5_1_transformer_blocks_0_ff_net_0_proj(self.input_blocks_5_1_transformer_blocks_0_norm3(v_303)), chunks=2, dim=-1)
        v_315 = v_255+ self.input_blocks_5_1_proj_out((self.input_blocks_5_1_transformer_blocks_0_ff_net_2(v_306 * F.gelu(input=v_307, approximate='none')) + v_303).reshape(k70, k73a, k74a, 640).permute(0,3,1,2))
#===
        v_316 = self.input_blocks_6_0_op(v_315)

        v_332 = self.input_blocks_7_0_skip_connection(v_316) + self.input_blocks_7_0_out_layers_3(F.silu(self.input_blocks_7_0_out_layers_0(self.input_blocks_7_0_in_layers_2(F.silu(self.input_blocks_7_0_in_layers_0(v_316))) + self.input_blocks_7_0_emb_layers_1(v_18).unsqueeze(2).unsqueeze(3))))

        v_336 = self.input_blocks_7_1_proj_in(self.input_blocks_7_1_norm(v_332)).permute(0,2,3,1).reshape(k70, -1, 1280)
        v_337 = self.input_blocks_7_1_transformer_blocks_0_norm1(v_336)

        v_347 = self.input_blocks_7_1_transformer_blocks_0_attn1_to_q(v_337).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_345 = self.input_blocks_7_1_transformer_blocks_0_attn1_to_k(v_337).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_352 = self.input_blocks_7_1_transformer_blocks_0_attn1_to_v(v_337).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_356 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_347, v_345) * 7.905694e-02), dim=-1), v_352).reshape(k70, 8, -1, 160).permute(0,2,1,3).reshape(k70, -1, 1280)
        v_358 = (self.input_blocks_7_1_transformer_blocks_0_attn1_to_out_0(v_356) + v_336)

        v_369 = self.input_blocks_7_1_transformer_blocks_0_attn2_to_q(self.input_blocks_7_1_transformer_blocks_0_norm2(v_358)).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_367 = self.input_blocks_7_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_374 = self.input_blocks_7_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_378 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_369, v_367) * 7.905694e-02), dim=-1), v_374).reshape(k70, 8, -1, 160).permute(0,2,1,3).reshape(k70, -1, 1280)
        v_380 = (self.input_blocks_7_1_transformer_blocks_0_attn2_to_out_0(v_378) + v_358)

        k73b=k73a>>1
        k74b=k74a>>1
        v_383, v_384 = torch.chunk(input=self.input_blocks_7_1_transformer_blocks_0_ff_net_0_proj(self.input_blocks_7_1_transformer_blocks_0_norm3(v_380)), chunks=2, dim=-1)
        v_394 = v_332+ self.input_blocks_7_1_proj_out((self.input_blocks_7_1_transformer_blocks_0_ff_net_2(v_383 * F.gelu(input=v_384, approximate='none')) + v_380).reshape(k70, k73b, k74b, 1280).permute(0,3,1,2))
#===
        v_407 = v_394 + self.input_blocks_8_0_out_layers_3(F.silu(self.input_blocks_8_0_out_layers_0(self.input_blocks_8_0_in_layers_2(F.silu(self.input_blocks_8_0_in_layers_0(v_394))) + self.input_blocks_8_0_emb_layers_1(v_18).unsqueeze(2).unsqueeze(3))))

        v_411 = self.input_blocks_8_1_proj_in(self.input_blocks_8_1_norm(v_407)).permute(0,2,3,1).reshape(k70, -1, 1280)
        v_412 = self.input_blocks_8_1_transformer_blocks_0_norm1(v_411)

        v_422 = self.input_blocks_8_1_transformer_blocks_0_attn1_to_q(v_412).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_420 = self.input_blocks_8_1_transformer_blocks_0_attn1_to_k(v_412).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_427 = self.input_blocks_8_1_transformer_blocks_0_attn1_to_v(v_412).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_431 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_422, v_420) * 7.905694e-02), dim=-1), v_427).reshape(k70, 8, -1, 160).permute(0,2,1,3).reshape(k70, -1, 1280)
        v_433 = (self.input_blocks_8_1_transformer_blocks_0_attn1_to_out_0(v_431) + v_411)

        v_444 = self.input_blocks_8_1_transformer_blocks_0_attn2_to_q(self.input_blocks_8_1_transformer_blocks_0_norm2(v_433)).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_442 = self.input_blocks_8_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_449 = self.input_blocks_8_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_453 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_444, v_442) * 7.905694e-02), dim=-1), v_449).reshape(k70, 8, -1, 160).permute(0,2,1,3).reshape(k70, -1, 1280)
        v_455 = (self.input_blocks_8_1_transformer_blocks_0_attn2_to_out_0(v_453) + v_433)

        v_458, v_459 = torch.chunk(input=self.input_blocks_8_1_transformer_blocks_0_ff_net_0_proj(self.input_blocks_8_1_transformer_blocks_0_norm3(v_455)), chunks=2, dim=-1)
        v_467 = v_407 + self.input_blocks_8_1_proj_out((self.input_blocks_8_1_transformer_blocks_0_ff_net_2(v_458 * F.gelu(input=v_459, approximate='none')) + v_455).reshape(k70, k73b, k74b, 1280).permute(0,3,1,2))
#===
        v_468 = self.input_blocks_9_0_op(v_467)

        v_485 = v_468 + self.input_blocks_10_0_out_layers_3(F.silu(self.input_blocks_10_0_out_layers_0(self.input_blocks_10_0_in_layers_2(F.silu(self.input_blocks_10_0_in_layers_0(v_468))) + self.input_blocks_10_0_emb_layers_1(v_18).unsqueeze(2).unsqueeze(3))))

        v_500 = v_485 + self.input_blocks_11_0_out_layers_3(F.silu(self.input_blocks_11_0_out_layers_0(self.input_blocks_11_0_in_layers_2(F.silu(self.input_blocks_11_0_in_layers_0(v_485))) + self.input_blocks_11_0_emb_layers_1(v_18).unsqueeze(2).unsqueeze(3))))
#===
        v_513 = v_500 + self.middle_block_0_out_layers_3(F.silu(self.middle_block_0_out_layers_0(self.middle_block_0_in_layers_2(F.silu(self.middle_block_0_in_layers_0(v_500))) + self.middle_block_0_emb_layers_1(v_18).unsqueeze(2).unsqueeze(3))))

        v_517 = self.middle_block_1_proj_in(self.middle_block_1_norm(v_513)).permute(0,2,3,1).reshape(k70, -1, 1280)
        v_518 = self.middle_block_1_transformer_blocks_0_norm1(v_517)

        v_528 = self.middle_block_1_transformer_blocks_0_attn1_to_q(v_518).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_526 = self.middle_block_1_transformer_blocks_0_attn1_to_k(v_518).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_533 = self.middle_block_1_transformer_blocks_0_attn1_to_v(v_518).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_537 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_528, v_526) * 7.905694e-02), dim=-1), v_533).reshape(k70, 8, -1, 160).permute(0,2,1,3).reshape(k70, -1, 1280)
        v_539 = (self.middle_block_1_transformer_blocks_0_attn1_to_out_0(v_537) + v_517)

        v_550 = self.middle_block_1_transformer_blocks_0_attn2_to_q(self.middle_block_1_transformer_blocks_0_norm2(v_539)).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_548 = self.middle_block_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_555 = self.middle_block_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_559 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_550, v_548) * 7.905694e-02), dim=-1), v_555).reshape(k70, 8, -1, 160).permute(0,2,1,3).reshape(k70, -1, 1280)
        v_561 = (self.middle_block_1_transformer_blocks_0_attn2_to_out_0(v_559) + v_539)

        k73c=k73b>>1
        k74c=k74b>>1
        v_564, v_565 = torch.chunk(input=self.middle_block_1_transformer_blocks_0_ff_net_0_proj(self.middle_block_1_transformer_blocks_0_norm3(v_561)), chunks=2, dim=-1)
        v_575 = v_513 +self.middle_block_1_proj_out((self.middle_block_1_transformer_blocks_0_ff_net_2(v_564 * F.gelu(input=v_565, approximate='none')) + v_561).reshape(k70, k73c, k74c, 1280).permute(0,3,1,2))

        v_588 = v_575 + self.middle_block_2_out_layers_3(F.silu(self.middle_block_2_out_layers_0(self.middle_block_2_in_layers_2(F.silu(self.middle_block_2_in_layers_0(v_575))) + self.middle_block_2_emb_layers_1(v_18).unsqueeze(2).unsqueeze(3))))

        v_589 = [v_12, v_90, v_163, v_164, v_242, v_315, v_316, v_394, v_467, v_468, v_485, v_500]
        return (v_588, v_11, v_589)


with init_empty_weights():
    jitbase.diffusion_emb_base= diffusion_emb().requires_grad_(False).eval()