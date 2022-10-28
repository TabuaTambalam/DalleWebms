import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
import jitbase

torch.set_grad_enabled(False)

class diffusion_mid(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_blocks_0_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=2560,eps=0.000010)

        self.output_blocks_0_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2560, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))

        self.output_blocks_0_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.output_blocks_0_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.output_blocks_0_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_0_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2560, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_1_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=2560,eps=0.000010)

        self.output_blocks_1_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2560, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_1_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.output_blocks_1_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.output_blocks_1_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_1_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2560, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_2_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=2560,eps=0.000010)

        self.output_blocks_2_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2560, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_2_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.output_blocks_2_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.output_blocks_2_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_2_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2560, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_2_1_conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_3_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=2560,eps=0.000010)

        self.output_blocks_3_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2560, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_3_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.output_blocks_3_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.output_blocks_3_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_3_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2560, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_3_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=1280, num_groups=32)
        self.output_blocks_3_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_3_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.output_blocks_3_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=1280, out_features=1280)
        self.output_blocks_3_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=1280, out_features=1280)
        self.output_blocks_3_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=1280, out_features=1280)
        self.output_blocks_3_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.output_blocks_3_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.output_blocks_3_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=1280, out_features=1280)
        self.output_blocks_3_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=1280)
        self.output_blocks_3_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=1280)
        self.output_blocks_3_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.output_blocks_3_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.output_blocks_3_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=1280, out_features=10240)
        self.output_blocks_3_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=5120, out_features=1280)
        self.output_blocks_3_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_4_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=2560,eps=0.000010)

        self.output_blocks_4_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2560, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_4_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.output_blocks_4_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.output_blocks_4_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_4_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2560, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_4_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=1280, num_groups=32)
        self.output_blocks_4_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_4_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.output_blocks_4_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=1280, out_features=1280)
        self.output_blocks_4_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=1280, out_features=1280)
        self.output_blocks_4_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=1280, out_features=1280)
        self.output_blocks_4_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.output_blocks_4_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.output_blocks_4_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=1280, out_features=1280)
        self.output_blocks_4_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=1280)
        self.output_blocks_4_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=1280)
        self.output_blocks_4_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.output_blocks_4_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.output_blocks_4_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=1280, out_features=10240)
        self.output_blocks_4_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=5120, out_features=1280)
        self.output_blocks_4_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_5_0_in_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1920,eps=0.000010)

        self.output_blocks_5_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1920, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_5_0_emb_layers_1 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.output_blocks_5_0_out_layers_0 = nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.output_blocks_5_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_5_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1920, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_5_1_norm = nn.GroupNorm(affine=True, eps=0.000001, num_channels=1280, num_groups=32)
        self.output_blocks_5_1_proj_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_5_1_transformer_blocks_0_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.output_blocks_5_1_transformer_blocks_0_attn1_to_q = nn.Linear(bias=False, in_features=1280, out_features=1280)
        self.output_blocks_5_1_transformer_blocks_0_attn1_to_k = nn.Linear(bias=jitbase.kvbias, in_features=1280, out_features=1280)
        self.output_blocks_5_1_transformer_blocks_0_attn1_to_v = nn.Linear(bias=jitbase.kvbias, in_features=1280, out_features=1280)
        self.output_blocks_5_1_transformer_blocks_0_attn1_to_out_0 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.output_blocks_5_1_transformer_blocks_0_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.output_blocks_5_1_transformer_blocks_0_attn2_to_q = nn.Linear(bias=False, in_features=1280, out_features=1280)
        self.output_blocks_5_1_transformer_blocks_0_attn2_to_k = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=1280)
        self.output_blocks_5_1_transformer_blocks_0_attn2_to_v = nn.Linear(bias=jitbase.kvbias, in_features=768, out_features=1280)
        self.output_blocks_5_1_transformer_blocks_0_attn2_to_out_0 = nn.Linear(bias=True, in_features=1280, out_features=1280)
        self.output_blocks_5_1_transformer_blocks_0_norm3 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(1280,))
        self.output_blocks_5_1_transformer_blocks_0_ff_net_0_proj = nn.Linear(bias=True, in_features=1280, out_features=10240)
        self.output_blocks_5_1_transformer_blocks_0_ff_net_2 = nn.Linear(bias=True, in_features=5120, out_features=1280)
        self.output_blocks_5_1_proj_out = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=1280, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_5_2_conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=1280, padding=(1,1), padding_mode='zeros', stride=(1,1))

    

    def forward(self, v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8):
        k77=v_2.size(1)
        k70 = v_0.size(0)
        k73 = v_0.size(2)<<1
        k74 = v_0.size(3)<<1
        k70x8=k70<<3

        v_11 = torch.cat((v_0, v_8), dim=1)
        v_15=F.silu(v_1)

        v_29 = torch.cat((self.output_blocks_0_0_skip_connection(v_11) + self.output_blocks_0_0_out_layers_3(F.silu(self.output_blocks_0_0_out_layers_0 (self.output_blocks_0_0_in_layers_2(F.silu(self.output_blocks_0_0_in_layers_0(v_11))) + self.output_blocks_0_0_emb_layers_1(v_15).unsqueeze(2).unsqueeze(3)))), v_7), dim=1)

        v_46 = torch.cat((self.output_blocks_1_0_skip_connection(v_29) + self.output_blocks_1_0_out_layers_3(F.silu(self.output_blocks_1_0_out_layers_0(self.output_blocks_1_0_in_layers_2(F.silu(self.output_blocks_1_0_in_layers_0(v_29))) + self.output_blocks_1_0_emb_layers_1(v_15).unsqueeze(2).unsqueeze(3)))), v_6), dim=1)

        v_60 = self.output_blocks_2_0_skip_connection(v_46) + self.output_blocks_2_0_out_layers_3(F.silu(self.output_blocks_2_0_out_layers_0(self.output_blocks_2_0_in_layers_2(F.silu(self.output_blocks_2_0_in_layers_0(v_46))) + self.output_blocks_2_0_emb_layers_1(v_15).unsqueeze(2).unsqueeze(3))))
        v_65 = torch.cat((self.output_blocks_2_1_conv(F.interpolate(input=v_60 , scale_factor=2.0, mode='nearest')), v_5), dim=1)

#===
        v_79 = (self.output_blocks_3_0_skip_connection(v_65) + self.output_blocks_3_0_out_layers_3(F.silu(self.output_blocks_3_0_out_layers_0(self.output_blocks_3_0_in_layers_2(F.silu(self.output_blocks_3_0_in_layers_0(v_65))) + self.output_blocks_3_0_emb_layers_1(v_15).unsqueeze(2).unsqueeze(3)))))
 
        v_83 = self.output_blocks_3_1_proj_in(self.output_blocks_3_1_norm(v_79)).permute(0,2,3,1).reshape(k70, -1, 1280)
        v_84 = self.output_blocks_3_1_transformer_blocks_0_norm1(v_83)

        v_94 = self.output_blocks_3_1_transformer_blocks_0_attn1_to_q(v_84).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_92 = self.output_blocks_3_1_transformer_blocks_0_attn1_to_k(v_84).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_99 = self.output_blocks_3_1_transformer_blocks_0_attn1_to_v(v_84).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_103 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_94, v_92) * 7.905694e-02), dim=-1), v_99).reshape(k70, 8, -1, 160).permute(0,2,1,3).reshape(k70, -1, 1280)
        v_105 = (self.output_blocks_3_1_transformer_blocks_0_attn1_to_out_0(v_103) + v_83)

        v_116 = self.output_blocks_3_1_transformer_blocks_0_attn2_to_q(self.output_blocks_3_1_transformer_blocks_0_norm2(v_105)).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_114 = self.output_blocks_3_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, k77, 8, 160).permute(0,2,1,3).reshape(k70x8, k77, 160)
        v_121 = self.output_blocks_3_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, k77, 8, 160).permute(0,2,1,3).reshape(k70x8, k77, 160)
        v_125 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_116, v_114) * 7.905694e-02), dim=-1), v_121).reshape(k70, 8, -1, 160).permute(0,2,1,3).reshape(k70, -1, 1280)
        v_127 = (self.output_blocks_3_1_transformer_blocks_0_attn2_to_out_0(v_125) + v_105)

        v_130, v_131 = torch.chunk(input=self.output_blocks_3_1_transformer_blocks_0_ff_net_0_proj(self.output_blocks_3_1_transformer_blocks_0_norm3(v_127)), chunks=2, dim=-1)
        v_142 = torch.cat((v_79+ self.output_blocks_3_1_proj_out((v_127+ self.output_blocks_3_1_transformer_blocks_0_ff_net_2(v_130 * F.gelu(input=v_131, approximate='none'))).reshape(k70, k73, k74, 1280).permute(0,3,1,2)), v_4), dim=1)
#===
        v_156 = self.output_blocks_4_0_skip_connection(v_142) + self.output_blocks_4_0_out_layers_3(F.silu(self.output_blocks_4_0_out_layers_0(self.output_blocks_4_0_in_layers_2(F.silu(self.output_blocks_4_0_in_layers_0(v_142))) + self.output_blocks_4_0_emb_layers_1(v_15).unsqueeze(2).unsqueeze(3))))

        v_160 = self.output_blocks_4_1_proj_in(self.output_blocks_4_1_norm(v_156)).permute(0,2,3,1).reshape(k70, -1, 1280)
        v_161 = self.output_blocks_4_1_transformer_blocks_0_norm1(v_160)

        v_171 = self.output_blocks_4_1_transformer_blocks_0_attn1_to_q(v_161).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_169 = self.output_blocks_4_1_transformer_blocks_0_attn1_to_k(v_161).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_176 = self.output_blocks_4_1_transformer_blocks_0_attn1_to_v(v_161).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_180 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_171, v_169) * 7.905694e-02), dim=-1), v_176).reshape(k70, 8, -1, 160).permute(0,2,1,3).reshape(k70, -1, 1280)
        v_182 = (self.output_blocks_4_1_transformer_blocks_0_attn1_to_out_0(v_180) + v_160)

        v_193 = self.output_blocks_4_1_transformer_blocks_0_attn2_to_q(self.output_blocks_4_1_transformer_blocks_0_norm2(v_182)).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_191 = self.output_blocks_4_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, k77, 8, 160).permute(0,2,1,3).reshape(k70x8, k77, 160)
        v_198 = self.output_blocks_4_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, k77, 8, 160).permute(0,2,1,3).reshape(k70x8, k77, 160)
        v_202 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_193, v_191) * 7.905694e-02), dim=-1), v_198).reshape(k70, 8, -1, 160).permute(0,2,1,3).reshape(k70, -1, 1280)
        v_204 = (self.output_blocks_4_1_transformer_blocks_0_attn2_to_out_0(v_202) + v_182)

        v_207, v_208 = torch.chunk(input=self.output_blocks_4_1_transformer_blocks_0_ff_net_0_proj(self.output_blocks_4_1_transformer_blocks_0_norm3(v_204)), chunks=2, dim=-1)
        v_219 = torch.cat((v_156+ self.output_blocks_4_1_proj_out((self.output_blocks_4_1_transformer_blocks_0_ff_net_2(v_207 * F.gelu(input=v_208, approximate='none')) + v_204).reshape(k70, k73, k74, 1280).permute(0,3,1,2)), v_3), dim=1)
#===
        v_233 = self.output_blocks_5_0_skip_connection(v_219) + self.output_blocks_5_0_out_layers_3(F.silu(self.output_blocks_5_0_out_layers_0 (self.output_blocks_5_0_in_layers_2(F.silu(self.output_blocks_5_0_in_layers_0(v_219))) + self.output_blocks_5_0_emb_layers_1(v_15).unsqueeze(2).unsqueeze(3))))
 
        v_237 = self.output_blocks_5_1_proj_in(self.output_blocks_5_1_norm(v_233)).permute(0,2,3,1).reshape(k70, -1, 1280)
        v_238 = self.output_blocks_5_1_transformer_blocks_0_norm1(v_237)

        v_248 = self.output_blocks_5_1_transformer_blocks_0_attn1_to_q(v_238).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_246 = self.output_blocks_5_1_transformer_blocks_0_attn1_to_k(v_238).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_253 = self.output_blocks_5_1_transformer_blocks_0_attn1_to_v(v_238).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_257 =  torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_248, v_246) * 7.905694e-02), dim=-1), v_253).reshape(k70, 8, -1, 160).permute(0,2,1,3).reshape(k70, -1, 1280)
        v_259 = (self.output_blocks_5_1_transformer_blocks_0_attn1_to_out_0(v_257) + v_237)

        v_270 = self.output_blocks_5_1_transformer_blocks_0_attn2_to_q(self.output_blocks_5_1_transformer_blocks_0_norm2(v_259)).reshape(k70, -1, 8, 160).permute(0,2,1,3).reshape(k70x8, -1, 160)
        v_268 = self.output_blocks_5_1_transformer_blocks_0_attn2_to_k(v_2).reshape(k70, k77, 8, 160).permute(0,2,1,3).reshape(k70x8, k77, 160)
        v_275 = self.output_blocks_5_1_transformer_blocks_0_attn2_to_v(v_2).reshape(k70, k77, 8, 160).permute(0,2,1,3).reshape(k70x8, k77, 160)
        v_279 = torch.einsum('i j l, i l k -> i j k', F.softmax(input=(torch.einsum('i j l, i k l -> i j k', v_270, v_268) * 7.905694e-02), dim=-1), v_275).reshape(k70, 8, -1, 160).permute(0,2,1,3).reshape(k70, -1, 1280)
        v_281 = (self.output_blocks_5_1_transformer_blocks_0_attn2_to_out_0(v_279) + v_259)

        v_284, v_285 = torch.chunk(input=self.output_blocks_5_1_transformer_blocks_0_ff_net_0_proj(self.output_blocks_5_1_transformer_blocks_0_norm3(v_281)), chunks=2, dim=-1)
        v_293 = v_233+ self.output_blocks_5_1_proj_out((self.output_blocks_5_1_transformer_blocks_0_ff_net_2(v_284 * F.gelu(input=v_285, approximate='none')) + v_281).reshape(k70, k73, k74, 1280).permute(0,3,1,2))

        return self.output_blocks_5_2_conv(F.interpolate(input=v_293 , scale_factor=2.0, mode='nearest'))


with init_empty_weights():
    jitbase.diffusion_mid_base= diffusion_mid().requires_grad_(False).eval()