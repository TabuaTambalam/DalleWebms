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
        self.token_embedding = nn.Embedding(embedding_dim=1024, num_embeddings=49408, sparse=False)
        self.tfrb_0_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_0_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_0_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_0_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_0_mlp_gelu = nn.GELU()
        self.tfrb_0_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_1_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_1_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_1_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_1_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_1_mlp_gelu = nn.GELU()
        self.tfrb_1_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_2_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_2_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_2_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_2_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_2_mlp_gelu = nn.GELU()
        self.tfrb_2_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_3_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_3_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_3_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_3_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_3_mlp_gelu = nn.GELU()
        self.tfrb_3_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_4_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_4_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_4_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_4_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_4_mlp_gelu = nn.GELU()
        self.tfrb_4_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_5_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_5_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_5_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_5_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_5_mlp_gelu = nn.GELU()
        self.tfrb_5_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_6_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_6_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_6_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_6_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_6_mlp_gelu = nn.GELU()
        self.tfrb_6_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_7_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_7_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_7_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_7_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_7_mlp_gelu = nn.GELU()
        self.tfrb_7_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_8_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_8_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_8_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_8_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_8_mlp_gelu = nn.GELU()
        self.tfrb_8_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_9_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_9_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_9_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_9_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_9_mlp_gelu = nn.GELU()
        self.tfrb_9_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_10_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_10_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_10_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_10_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_10_mlp_gelu = nn.GELU()
        self.tfrb_10_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_11_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_11_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_11_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_11_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_11_mlp_gelu = nn.GELU()
        self.tfrb_11_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_12_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_12_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_12_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_12_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_12_mlp_gelu = nn.GELU()
        self.tfrb_12_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_13_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_13_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_13_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_13_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_13_mlp_gelu = nn.GELU()
        self.tfrb_13_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_14_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_14_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_14_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_14_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_14_mlp_gelu = nn.GELU()
        self.tfrb_14_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_15_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_15_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_15_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_15_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_15_mlp_gelu = nn.GELU()
        self.tfrb_15_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_16_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_16_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_16_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_16_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_16_mlp_gelu = nn.GELU()
        self.tfrb_16_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_17_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_17_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_17_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_17_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_17_mlp_gelu = nn.GELU()
        self.tfrb_17_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_18_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_18_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_18_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_18_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_18_mlp_gelu = nn.GELU()
        self.tfrb_18_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_19_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_19_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_19_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_19_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_19_mlp_gelu = nn.GELU()
        self.tfrb_19_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_20_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_20_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_20_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_20_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_20_mlp_gelu = nn.GELU()
        self.tfrb_20_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_21_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_21_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_21_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_21_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_21_mlp_gelu = nn.GELU()
        self.tfrb_21_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_22_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_22_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_22_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_22_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_22_mlp_gelu = nn.GELU()
        self.tfrb_22_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)

        self.tfrb_23_ln_1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_23_ln_2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.tfrb_23_attn = nn.MultiheadAttention(add_bias_kv=False, add_zero_attn=False, batch_first=False, bias=True, embed_dim=1024, num_heads=16)
        self.tfrb_23_mlp_c_fc = nn.Linear(bias=True, in_features=1024, out_features=4096)
        self.tfrb_23_mlp_gelu = nn.GELU()
        self.tfrb_23_mlp_c_proj = nn.Linear(bias=True, in_features=4096, out_features=1024)
        self.ln_final = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1024,))
        self.text_projection = nn.Parameter(torch.ones(1), requires_grad=False)
        self.attn_mask = nn.Parameter(torch.ones(1), requires_grad=False)
        self.positional_embedding = nn.Parameter(torch.ones(1), requires_grad=False)


    def embedding(self, x):
        return self.token_embedding(x)

    def forward(self, x):
        return self.ln_final(self.forward2(x).permute(1,0,2))

    def forward2(self, v_0):
        attn_mask = self.attn_mask

        LastO = (v_0 + self.positional_embedding).permute(1,0,2)

        norm1 =self.tfrb_0_ln_1(LastO)
        at1, _ = self.tfrb_0_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_0_mlp_c_proj(self.tfrb_0_mlp_gelu(self.tfrb_0_mlp_c_fc(self.tfrb_0_ln_2(at2)))))

        norm1 =self.tfrb_1_ln_1(LastO)
        at1, _ = self.tfrb_1_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_1_mlp_c_proj(self.tfrb_1_mlp_gelu(self.tfrb_1_mlp_c_fc(self.tfrb_1_ln_2(at2)))))

        norm1 =self.tfrb_2_ln_1(LastO)
        at1, _ = self.tfrb_2_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_2_mlp_c_proj(self.tfrb_2_mlp_gelu(self.tfrb_2_mlp_c_fc(self.tfrb_2_ln_2(at2)))))

        norm1 =self.tfrb_3_ln_1(LastO)
        at1, _ = self.tfrb_3_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_3_mlp_c_proj(self.tfrb_3_mlp_gelu(self.tfrb_3_mlp_c_fc(self.tfrb_3_ln_2(at2)))))

        norm1 =self.tfrb_4_ln_1(LastO)
        at1, _ = self.tfrb_4_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_4_mlp_c_proj(self.tfrb_4_mlp_gelu(self.tfrb_4_mlp_c_fc(self.tfrb_4_ln_2(at2)))))

        norm1 =self.tfrb_5_ln_1(LastO)
        at1, _ = self.tfrb_5_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_5_mlp_c_proj(self.tfrb_5_mlp_gelu(self.tfrb_5_mlp_c_fc(self.tfrb_5_ln_2(at2)))))

        norm1 =self.tfrb_6_ln_1(LastO)
        at1, _ = self.tfrb_6_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_6_mlp_c_proj(self.tfrb_6_mlp_gelu(self.tfrb_6_mlp_c_fc(self.tfrb_6_ln_2(at2)))))

        norm1 =self.tfrb_7_ln_1(LastO)
        at1, _ = self.tfrb_7_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_7_mlp_c_proj(self.tfrb_7_mlp_gelu(self.tfrb_7_mlp_c_fc(self.tfrb_7_ln_2(at2)))))

        norm1 =self.tfrb_8_ln_1(LastO)
        at1, _ = self.tfrb_8_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_8_mlp_c_proj(self.tfrb_8_mlp_gelu(self.tfrb_8_mlp_c_fc(self.tfrb_8_ln_2(at2)))))

        norm1 =self.tfrb_9_ln_1(LastO)
        at1, _ = self.tfrb_9_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_9_mlp_c_proj(self.tfrb_9_mlp_gelu(self.tfrb_9_mlp_c_fc(self.tfrb_9_ln_2(at2)))))

        norm1 =self.tfrb_10_ln_1(LastO)
        at1, _ = self.tfrb_10_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_10_mlp_c_proj(self.tfrb_10_mlp_gelu(self.tfrb_10_mlp_c_fc(self.tfrb_10_ln_2(at2)))))

        norm1 =self.tfrb_11_ln_1(LastO)
        at1, _ = self.tfrb_11_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_11_mlp_c_proj(self.tfrb_11_mlp_gelu(self.tfrb_11_mlp_c_fc(self.tfrb_11_ln_2(at2)))))

        norm1 =self.tfrb_12_ln_1(LastO)
        at1, _ = self.tfrb_12_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_12_mlp_c_proj(self.tfrb_12_mlp_gelu(self.tfrb_12_mlp_c_fc(self.tfrb_12_ln_2(at2)))))

        norm1 =self.tfrb_13_ln_1(LastO)
        at1, _ = self.tfrb_13_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_13_mlp_c_proj(self.tfrb_13_mlp_gelu(self.tfrb_13_mlp_c_fc(self.tfrb_13_ln_2(at2)))))

        norm1 =self.tfrb_14_ln_1(LastO)
        at1, _ = self.tfrb_14_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_14_mlp_c_proj(self.tfrb_14_mlp_gelu(self.tfrb_14_mlp_c_fc(self.tfrb_14_ln_2(at2)))))

        norm1 =self.tfrb_15_ln_1(LastO)
        at1, _ = self.tfrb_15_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_15_mlp_c_proj(self.tfrb_15_mlp_gelu(self.tfrb_15_mlp_c_fc(self.tfrb_15_ln_2(at2)))))

        norm1 =self.tfrb_16_ln_1(LastO)
        at1, _ = self.tfrb_16_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_16_mlp_c_proj(self.tfrb_16_mlp_gelu(self.tfrb_16_mlp_c_fc(self.tfrb_16_ln_2(at2)))))

        norm1 =self.tfrb_17_ln_1(LastO)
        at1, _ = self.tfrb_17_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_17_mlp_c_proj(self.tfrb_17_mlp_gelu(self.tfrb_17_mlp_c_fc(self.tfrb_17_ln_2(at2)))))

        norm1 =self.tfrb_18_ln_1(LastO)
        at1, _ = self.tfrb_18_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_18_mlp_c_proj(self.tfrb_18_mlp_gelu(self.tfrb_18_mlp_c_fc(self.tfrb_18_ln_2(at2)))))

        norm1 =self.tfrb_19_ln_1(LastO)
        at1, _ = self.tfrb_19_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_19_mlp_c_proj(self.tfrb_19_mlp_gelu(self.tfrb_19_mlp_c_fc(self.tfrb_19_ln_2(at2)))))
        if self.cut == -4:
          return LastO
        norm1 =self.tfrb_20_ln_1(LastO)
        at1, _ = self.tfrb_20_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_20_mlp_c_proj(self.tfrb_20_mlp_gelu(self.tfrb_20_mlp_c_fc(self.tfrb_20_ln_2(at2)))))
        if self.cut == -3:
          return LastO
        norm1 =self.tfrb_21_ln_1(LastO)
        at1, _ = self.tfrb_21_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_21_mlp_c_proj(self.tfrb_21_mlp_gelu(self.tfrb_21_mlp_c_fc(self.tfrb_21_ln_2(at2)))))
        if self.cut == -2:
          return LastO
        norm1 =self.tfrb_22_ln_1(LastO)
        at1, _ = self.tfrb_22_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_22_mlp_c_proj(self.tfrb_22_mlp_gelu(self.tfrb_22_mlp_c_fc(self.tfrb_22_ln_2(at2)))))
        if self.cut == -1:
          return LastO
        norm1 =self.tfrb_23_ln_1(LastO)
        at1, _ = self.tfrb_23_attn(norm1,norm1,norm1,attn_mask=attn_mask)
        at2 = (LastO + at1)
        LastO = (at2 + self.tfrb_23_mlp_c_proj(self.tfrb_23_mlp_gelu(self.tfrb_23_mlp_c_fc(self.tfrb_23_ln_2(at2)))))

        return LastO


with init_empty_weights():
    jitbase= Model().requires_grad_(False).eval()
