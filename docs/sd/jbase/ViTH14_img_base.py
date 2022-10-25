import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
torch.set_grad_enabled(False)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cut=0
        self.visual_conv1 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=3, kernel_size=(14,14), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(14,14))
        self.tfrb_0_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_0_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_0_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_0_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_0_mlp_gelu = nn.GELU()
        self.tfrb_0_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_1_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_1_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_1_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_1_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_1_mlp_gelu = nn.GELU()
        self.tfrb_1_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_2_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_2_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_2_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_2_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_2_mlp_gelu = nn.GELU()
        self.tfrb_2_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_3_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_3_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_3_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_3_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_3_mlp_gelu = nn.GELU()
        self.tfrb_3_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_4_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_4_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_4_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_4_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_4_mlp_gelu = nn.GELU()
        self.tfrb_4_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_5_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_5_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_5_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_5_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_5_mlp_gelu = nn.GELU()
        self.tfrb_5_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_6_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_6_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_6_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_6_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_6_mlp_gelu = nn.GELU()
        self.tfrb_6_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_7_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_7_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_7_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_7_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_7_mlp_gelu = nn.GELU()
        self.tfrb_7_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_8_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_8_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_8_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_8_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_8_mlp_gelu = nn.GELU()
        self.tfrb_8_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_9_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_9_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_9_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_9_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_9_mlp_gelu = nn.GELU()
        self.tfrb_9_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_10_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_10_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_10_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_10_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_10_mlp_gelu = nn.GELU()
        self.tfrb_10_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_11_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_11_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_11_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_11_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_11_mlp_gelu = nn.GELU()
        self.tfrb_11_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_12_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_12_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_12_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_12_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_12_mlp_gelu = nn.GELU()
        self.tfrb_12_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_13_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_13_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_13_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_13_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_13_mlp_gelu = nn.GELU()
        self.tfrb_13_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_14_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_14_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_14_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_14_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_14_mlp_gelu = nn.GELU()
        self.tfrb_14_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_15_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_15_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_15_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_15_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_15_mlp_gelu = nn.GELU()
        self.tfrb_15_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_16_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_16_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_16_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_16_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_16_mlp_gelu = nn.GELU()
        self.tfrb_16_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_17_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_17_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_17_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_17_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_17_mlp_gelu = nn.GELU()
        self.tfrb_17_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_18_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_18_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_18_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_18_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_18_mlp_gelu = nn.GELU()
        self.tfrb_18_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_19_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_19_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_19_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_19_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_19_mlp_gelu = nn.GELU()
        self.tfrb_19_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_20_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_20_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_20_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_20_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_20_mlp_gelu = nn.GELU()
        self.tfrb_20_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_21_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_21_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_21_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_21_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_21_mlp_gelu = nn.GELU()
        self.tfrb_21_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_22_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_22_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_22_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_22_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_22_mlp_gelu = nn.GELU()
        self.tfrb_22_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_23_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_23_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_23_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_23_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_23_mlp_gelu = nn.GELU()
        self.tfrb_23_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_24_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_24_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_24_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_24_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_24_mlp_gelu = nn.GELU()
        self.tfrb_24_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_25_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_25_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_25_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_25_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_25_mlp_gelu = nn.GELU()
        self.tfrb_25_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_26_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_26_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_26_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_26_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_26_mlp_gelu = nn.GELU()
        self.tfrb_26_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_27_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_27_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_27_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_27_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_27_mlp_gelu = nn.GELU()
        self.tfrb_27_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_28_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_28_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_28_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_28_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_28_mlp_gelu = nn.GELU()
        self.tfrb_28_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_29_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_29_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_29_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_29_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_29_mlp_gelu = nn.GELU()
        self.tfrb_29_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_30_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_30_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_30_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_30_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_30_mlp_gelu = nn.GELU()
        self.tfrb_30_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)

        self.tfrb_31_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_31_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.tfrb_31_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1280, num_heads=16)
        self.tfrb_31_mlp_c_fc = nn.Linear(bias=True, in_features=1280, out_features=5120)
        self.tfrb_31_mlp_gelu = nn.GELU()
        self.tfrb_31_mlp_c_proj = nn.Linear(bias=True, in_features=5120, out_features=1280)
        self.visual_ln_pre = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.visual_ln_post = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.visual_proj = nn.Parameter(torch.ones(1), requires_grad=False)
        self.positional_embedding = nn.Parameter(torch.ones(1), requires_grad=False)
        self.class_embedding = nn.Parameter(torch.ones(1), requires_grad=False)



    def forward(self,x):
        vpost = self.visual_ln_post(self.forward2(x).permute(1,0,2))
        #v_306 = torch.matmul(input=vpost.select(dim=1, index=0), other=self.visual_proj)
        return vpost

    def forward2(self, v_0):
        b=v_0.size(0)
        conv0 = self.visual_conv1(v_0).reshape(b, 1280, -1).permute(0,2,1)
        kla_conv0 = torch.cat((self.class_embedding.expand(b,1,-1), conv0), dim=1)
        LastO = self.visual_ln_pre(kla_conv0 + self.positional_embedding).permute(1,0,2)

        norm1 = self.tfrb_0_ln_1(LastO)
        at1, _ = self.tfrb_0_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_0_mlp_c_proj(self.tfrb_0_mlp_gelu(self.tfrb_0_mlp_c_fc(self.tfrb_0_ln_2(at2)))))

        norm1 = self.tfrb_1_ln_1(LastO)
        at1, _ = self.tfrb_1_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_1_mlp_c_proj(self.tfrb_1_mlp_gelu(self.tfrb_1_mlp_c_fc(self.tfrb_1_ln_2(at2)))))

        norm1 = self.tfrb_2_ln_1(LastO)
        at1, _ = self.tfrb_2_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_2_mlp_c_proj(self.tfrb_2_mlp_gelu(self.tfrb_2_mlp_c_fc(self.tfrb_2_ln_2(at2)))))

        norm1 = self.tfrb_3_ln_1(LastO)
        at1, _ = self.tfrb_3_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_3_mlp_c_proj(self.tfrb_3_mlp_gelu(self.tfrb_3_mlp_c_fc(self.tfrb_3_ln_2(at2)))))

        norm1 = self.tfrb_4_ln_1(LastO)
        at1, _ = self.tfrb_4_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_4_mlp_c_proj(self.tfrb_4_mlp_gelu(self.tfrb_4_mlp_c_fc(self.tfrb_4_ln_2(at2)))))

        norm1 = self.tfrb_5_ln_1(LastO)
        at1, _ = self.tfrb_5_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_5_mlp_c_proj(self.tfrb_5_mlp_gelu(self.tfrb_5_mlp_c_fc(self.tfrb_5_ln_2(at2)))))

        norm1 = self.tfrb_6_ln_1(LastO)
        at1, _ = self.tfrb_6_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_6_mlp_c_proj(self.tfrb_6_mlp_gelu(self.tfrb_6_mlp_c_fc(self.tfrb_6_ln_2(at2)))))

        norm1 = self.tfrb_7_ln_1(LastO)
        at1, _ = self.tfrb_7_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_7_mlp_c_proj(self.tfrb_7_mlp_gelu(self.tfrb_7_mlp_c_fc(self.tfrb_7_ln_2(at2)))))

        norm1 = self.tfrb_8_ln_1(LastO)
        at1, _ = self.tfrb_8_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_8_mlp_c_proj(self.tfrb_8_mlp_gelu(self.tfrb_8_mlp_c_fc(self.tfrb_8_ln_2(at2)))))

        norm1 = self.tfrb_9_ln_1(LastO)
        at1, _ = self.tfrb_9_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_9_mlp_c_proj(self.tfrb_9_mlp_gelu(self.tfrb_9_mlp_c_fc(self.tfrb_9_ln_2(at2)))))

        norm1 = self.tfrb_10_ln_1(LastO)
        at1, _ = self.tfrb_10_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_10_mlp_c_proj(self.tfrb_10_mlp_gelu(self.tfrb_10_mlp_c_fc(self.tfrb_10_ln_2(at2)))))

        norm1 = self.tfrb_11_ln_1(LastO)
        at1, _ = self.tfrb_11_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_11_mlp_c_proj(self.tfrb_11_mlp_gelu(self.tfrb_11_mlp_c_fc(self.tfrb_11_ln_2(at2)))))

        norm1 = self.tfrb_12_ln_1(LastO)
        at1, _ = self.tfrb_12_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_12_mlp_c_proj(self.tfrb_12_mlp_gelu(self.tfrb_12_mlp_c_fc(self.tfrb_12_ln_2(at2)))))

        norm1 = self.tfrb_13_ln_1(LastO)
        at1, _ = self.tfrb_13_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_13_mlp_c_proj(self.tfrb_13_mlp_gelu(self.tfrb_13_mlp_c_fc(self.tfrb_13_ln_2(at2)))))

        norm1 = self.tfrb_14_ln_1(LastO)
        at1, _ = self.tfrb_14_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_14_mlp_c_proj(self.tfrb_14_mlp_gelu(self.tfrb_14_mlp_c_fc(self.tfrb_14_ln_2(at2)))))

        norm1 = self.tfrb_15_ln_1(LastO)
        at1, _ = self.tfrb_15_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_15_mlp_c_proj(self.tfrb_15_mlp_gelu(self.tfrb_15_mlp_c_fc(self.tfrb_15_ln_2(at2)))))

        norm1 = self.tfrb_16_ln_1(LastO)
        at1, _ = self.tfrb_16_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_16_mlp_c_proj(self.tfrb_16_mlp_gelu(self.tfrb_16_mlp_c_fc(self.tfrb_16_ln_2(at2)))))

        norm1 = self.tfrb_17_ln_1(LastO)
        at1, _ = self.tfrb_17_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_17_mlp_c_proj(self.tfrb_17_mlp_gelu(self.tfrb_17_mlp_c_fc(self.tfrb_17_ln_2(at2)))))

        norm1 = self.tfrb_18_ln_1(LastO)
        at1, _ = self.tfrb_18_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_18_mlp_c_proj(self.tfrb_18_mlp_gelu(self.tfrb_18_mlp_c_fc(self.tfrb_18_ln_2(at2)))))

        norm1 = self.tfrb_19_ln_1(LastO)
        at1, _ = self.tfrb_19_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_19_mlp_c_proj(self.tfrb_19_mlp_gelu(self.tfrb_19_mlp_c_fc(self.tfrb_19_ln_2(at2)))))

        norm1 = self.tfrb_20_ln_1(LastO)
        at1, _ = self.tfrb_20_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_20_mlp_c_proj(self.tfrb_20_mlp_gelu(self.tfrb_20_mlp_c_fc(self.tfrb_20_ln_2(at2)))))

        norm1 = self.tfrb_21_ln_1(LastO)
        at1, _ = self.tfrb_21_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_21_mlp_c_proj(self.tfrb_21_mlp_gelu(self.tfrb_21_mlp_c_fc(self.tfrb_21_ln_2(at2)))))

        norm1 = self.tfrb_22_ln_1(LastO)
        at1, _ = self.tfrb_22_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_22_mlp_c_proj(self.tfrb_22_mlp_gelu(self.tfrb_22_mlp_c_fc(self.tfrb_22_ln_2(at2)))))

        norm1 = self.tfrb_23_ln_1(LastO)
        at1, _ = self.tfrb_23_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_23_mlp_c_proj(self.tfrb_23_mlp_gelu(self.tfrb_23_mlp_c_fc(self.tfrb_23_ln_2(at2)))))

        norm1 = self.tfrb_24_ln_1(LastO)
        at1, _ = self.tfrb_24_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_24_mlp_c_proj(self.tfrb_24_mlp_gelu(self.tfrb_24_mlp_c_fc(self.tfrb_24_ln_2(at2)))))

        norm1 = self.tfrb_25_ln_1(LastO)
        at1, _ = self.tfrb_25_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_25_mlp_c_proj(self.tfrb_25_mlp_gelu(self.tfrb_25_mlp_c_fc(self.tfrb_25_ln_2(at2)))))

        norm1 = self.tfrb_26_ln_1(LastO)
        at1, _ = self.tfrb_26_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_26_mlp_c_proj(self.tfrb_26_mlp_gelu(self.tfrb_26_mlp_c_fc(self.tfrb_26_ln_2(at2)))))

        norm1 = self.tfrb_27_ln_1(LastO)
        at1, _ = self.tfrb_27_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_27_mlp_c_proj(self.tfrb_27_mlp_gelu(self.tfrb_27_mlp_c_fc(self.tfrb_27_ln_2(at2)))))
        if self.cut == -4:
          return LastO
        norm1 = self.tfrb_28_ln_1(LastO)
        at1, _ = self.tfrb_28_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_28_mlp_c_proj(self.tfrb_28_mlp_gelu(self.tfrb_28_mlp_c_fc(self.tfrb_28_ln_2(at2)))))
        if self.cut == -3:
          return LastO
        norm1 = self.tfrb_29_ln_1(LastO)
        at1, _ = self.tfrb_29_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_29_mlp_c_proj(self.tfrb_29_mlp_gelu(self.tfrb_29_mlp_c_fc(self.tfrb_29_ln_2(at2)))))
        if self.cut == -2:
          return LastO
        norm1 = self.tfrb_30_ln_1(LastO)
        at1, _ = self.tfrb_30_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_30_mlp_c_proj(self.tfrb_30_mlp_gelu(self.tfrb_30_mlp_c_fc(self.tfrb_30_ln_2(at2)))))
        if self.cut == -1:
          return LastO
        norm1 = self.tfrb_31_ln_1(LastO)
        at1, _ = self.tfrb_31_attn(norm1, norm1, norm1)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_31_mlp_c_proj(self.tfrb_31_mlp_gelu(self.tfrb_31_mlp_c_fc(self.tfrb_31_ln_2(at2)))))

        return LastO

with init_empty_weights():
    jitbase= Model().requires_grad_(False).eval()