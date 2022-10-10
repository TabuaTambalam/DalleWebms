import torch
import torch.nn as nn
import torch.nn.functional as F
import jitbase

torch.set_grad_enabled(False)


class transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.cut=0
        self.text_model_embeddings_token_embedding = nn.Embedding(embedding_dim=768, num_embeddings=49408, sparse=False)
        self.text_model_encoder_layers_0_layer_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_0_self_attn_q_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_0_self_attn_k_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_0_self_attn_v_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_0_self_attn_out_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_0_layer_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_0_mlp_fc1 = nn.Linear(bias=True, in_features=768, out_features=3072)
        self.text_model_encoder_layers_0_mlp_fc2 = nn.Linear(bias=True, in_features=3072, out_features=768)
        self.text_model_encoder_layers_1_layer_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_1_self_attn_q_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_1_self_attn_k_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_1_self_attn_v_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_1_self_attn_out_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_1_layer_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_1_mlp_fc1 = nn.Linear(bias=True, in_features=768, out_features=3072)
        self.text_model_encoder_layers_1_mlp_fc2 = nn.Linear(bias=True, in_features=3072, out_features=768)
        self.text_model_encoder_layers_2_layer_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_2_self_attn_q_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_2_self_attn_k_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_2_self_attn_v_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_2_self_attn_out_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_2_layer_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_2_mlp_fc1 = nn.Linear(bias=True, in_features=768, out_features=3072)
        self.text_model_encoder_layers_2_mlp_fc2 = nn.Linear(bias=True, in_features=3072, out_features=768)
        self.text_model_encoder_layers_3_layer_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_3_self_attn_q_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_3_self_attn_k_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_3_self_attn_v_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_3_self_attn_out_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_3_layer_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_3_mlp_fc1 = nn.Linear(bias=True, in_features=768, out_features=3072)
        self.text_model_encoder_layers_3_mlp_fc2 = nn.Linear(bias=True, in_features=3072, out_features=768)
        self.text_model_encoder_layers_4_layer_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_4_self_attn_q_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_4_self_attn_k_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_4_self_attn_v_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_4_self_attn_out_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_4_layer_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_4_mlp_fc1 = nn.Linear(bias=True, in_features=768, out_features=3072)
        self.text_model_encoder_layers_4_mlp_fc2 = nn.Linear(bias=True, in_features=3072, out_features=768)
        self.text_model_encoder_layers_5_layer_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_5_self_attn_q_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_5_self_attn_k_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_5_self_attn_v_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_5_self_attn_out_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_5_layer_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_5_mlp_fc1 = nn.Linear(bias=True, in_features=768, out_features=3072)
        self.text_model_encoder_layers_5_mlp_fc2 = nn.Linear(bias=True, in_features=3072, out_features=768)
        self.text_model_encoder_layers_6_layer_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_6_self_attn_q_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_6_self_attn_k_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_6_self_attn_v_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_6_self_attn_out_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_6_layer_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_6_mlp_fc1 = nn.Linear(bias=True, in_features=768, out_features=3072)
        self.text_model_encoder_layers_6_mlp_fc2 = nn.Linear(bias=True, in_features=3072, out_features=768)
        self.text_model_encoder_layers_7_layer_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_7_self_attn_q_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_7_self_attn_k_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_7_self_attn_v_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_7_self_attn_out_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_7_layer_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_7_mlp_fc1 = nn.Linear(bias=True, in_features=768, out_features=3072)
        self.text_model_encoder_layers_7_mlp_fc2 = nn.Linear(bias=True, in_features=3072, out_features=768)
        self.text_model_encoder_layers_8_layer_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_8_self_attn_q_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_8_self_attn_k_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_8_self_attn_v_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_8_self_attn_out_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_8_layer_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_8_mlp_fc1 = nn.Linear(bias=True, in_features=768, out_features=3072)
        self.text_model_encoder_layers_8_mlp_fc2 = nn.Linear(bias=True, in_features=3072, out_features=768)
        self.text_model_encoder_layers_9_layer_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_9_self_attn_q_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_9_self_attn_k_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_9_self_attn_v_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_9_self_attn_out_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_9_layer_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_9_mlp_fc1 = nn.Linear(bias=True, in_features=768, out_features=3072)
        self.text_model_encoder_layers_9_mlp_fc2 = nn.Linear(bias=True, in_features=3072, out_features=768)
        self.text_model_encoder_layers_10_layer_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_10_self_attn_q_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_10_self_attn_k_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_10_self_attn_v_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_10_self_attn_out_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_10_layer_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_10_mlp_fc1 = nn.Linear(bias=True, in_features=768, out_features=3072)
        self.text_model_encoder_layers_10_mlp_fc2 = nn.Linear(bias=True, in_features=3072, out_features=768)
        self.text_model_encoder_layers_11_layer_norm1 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_11_self_attn_q_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_11_self_attn_k_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_11_self_attn_v_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_11_self_attn_out_proj = nn.Linear(bias=True, in_features=768, out_features=768)
        self.text_model_encoder_layers_11_layer_norm2 = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.text_model_encoder_layers_11_mlp_fc1 = nn.Linear(bias=True, in_features=768, out_features=3072)
        self.text_model_encoder_layers_11_mlp_fc2 = nn.Linear(bias=True, in_features=3072, out_features=768)
        self.text_model_final_layer_norm = nn.LayerNorm(elementwise_affine=True, eps=0.000010, normalized_shape=(768,))
        self.position_embedding=nn.Parameter(torch.ones(1), requires_grad=False)
        self.causal_attention_mask=nn.Parameter(torch.ones(1), requires_grad=False)

#np.load()[:77,:77]

    def embedding(self, v_0):
        return self.text_model_embeddings_token_embedding(v_0)

    def FinalNorm(self, v):
        return self.text_model_final_layer_norm(v)

    
    def forward(self, x_in):
      x=self.forward2(x_in)
      return self.FinalNorm(x)

    def forward2(self, v_0):

        k70 = v_0.size(0)
        k77 = v_0.size(1)
        k12=v_0.size(2)>>6
        k70x12=k70*k12

        causal_attention_mask = self.causal_attention_mask.expand([k70,-1,-1,-1])
        LastO = v_0 + self.position_embedding

        
        On1 = self.text_model_encoder_layers_0_layer_norm1(LastO)

        asQ = (self.text_model_encoder_layers_0_self_attn_q_proj(On1) /8).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)
        asK = self.text_model_encoder_layers_0_self_attn_k_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64).transpose(1,2)
        asV = self.text_model_encoder_layers_0_self_attn_v_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)

        bmmQK = (torch.bmm(input=asQ, mat2=asK).view(k70, k12, k77, k77)+ causal_attention_mask).view(k70x12, k77, k77)
        bmmQK = torch.softmax(input=bmmQK, dim=-1)

        bmmQKV = torch.bmm(input=bmmQK, mat2=asV).view(k70, k12, k77, 64).transpose(1,2).reshape(k70, k77, 768)

        asO = LastO+self.text_model_encoder_layers_0_self_attn_out_proj(bmmQKV)

        Omlp = self.text_model_encoder_layers_0_mlp_fc1(self.text_model_encoder_layers_0_layer_norm2(asO))
        LastO = asO + self.text_model_encoder_layers_0_mlp_fc2((Omlp * torch.sigmoid(input=Omlp * 1.702)))



        On1 = self.text_model_encoder_layers_1_layer_norm1(LastO)

        asQ = (self.text_model_encoder_layers_1_self_attn_q_proj(On1) /8).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)
        asK = self.text_model_encoder_layers_1_self_attn_k_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64).transpose(1,2)
        asV = self.text_model_encoder_layers_1_self_attn_v_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)

        bmmQK = (torch.bmm(input=asQ, mat2=asK).view(k70, k12, k77, k77)+ causal_attention_mask).view(k70x12, k77, k77)
        bmmQK = torch.softmax(input=bmmQK, dim=-1)

        bmmQKV = torch.bmm(input=bmmQK, mat2=asV).view(k70, k12, k77, 64).transpose(1,2).reshape(k70, k77, 768)

        asO = LastO+self.text_model_encoder_layers_1_self_attn_out_proj(bmmQKV)

        Omlp = self.text_model_encoder_layers_1_mlp_fc1(self.text_model_encoder_layers_1_layer_norm2(asO))
        LastO = asO + self.text_model_encoder_layers_1_mlp_fc2((Omlp * torch.sigmoid(input=Omlp * 1.702)))



        On1 = self.text_model_encoder_layers_2_layer_norm1(LastO)

        asQ = (self.text_model_encoder_layers_2_self_attn_q_proj(On1) /8).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)
        asK = self.text_model_encoder_layers_2_self_attn_k_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64).transpose(1,2)
        asV = self.text_model_encoder_layers_2_self_attn_v_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)

        bmmQK = (torch.bmm(input=asQ, mat2=asK).view(k70, k12, k77, k77)+ causal_attention_mask).view(k70x12, k77, k77)
        bmmQK = torch.softmax(input=bmmQK, dim=-1)

        bmmQKV = torch.bmm(input=bmmQK, mat2=asV).view(k70, k12, k77, 64).transpose(1,2).reshape(k70, k77, 768)

        asO = LastO+self.text_model_encoder_layers_2_self_attn_out_proj(bmmQKV)

        Omlp = self.text_model_encoder_layers_2_mlp_fc1(self.text_model_encoder_layers_2_layer_norm2(asO))
        LastO = asO + self.text_model_encoder_layers_2_mlp_fc2((Omlp * torch.sigmoid(input=Omlp * 1.702)))




        On1 = self.text_model_encoder_layers_3_layer_norm1(LastO)

        asQ = (self.text_model_encoder_layers_3_self_attn_q_proj(On1) /8).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)
        asK = self.text_model_encoder_layers_3_self_attn_k_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64).transpose(1,2)
        asV = self.text_model_encoder_layers_3_self_attn_v_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)

        bmmQK = (torch.bmm(input=asQ, mat2=asK).view(k70, k12, k77, k77)+ causal_attention_mask).view(k70x12, k77, k77)
        bmmQK = torch.softmax(input=bmmQK, dim=-1)

        bmmQKV = torch.bmm(input=bmmQK, mat2=asV).view(k70, k12, k77, 64).transpose(1,2).reshape(k70, k77, 768)

        asO = LastO+self.text_model_encoder_layers_3_self_attn_out_proj(bmmQKV)

        Omlp = self.text_model_encoder_layers_3_mlp_fc1(self.text_model_encoder_layers_3_layer_norm2(asO))
        LastO = asO + self.text_model_encoder_layers_3_mlp_fc2((Omlp * torch.sigmoid(input=Omlp * 1.702)))



        On1 = self.text_model_encoder_layers_4_layer_norm1(LastO)

        asQ = (self.text_model_encoder_layers_4_self_attn_q_proj(On1) /8).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)
        asK = self.text_model_encoder_layers_4_self_attn_k_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64).transpose(1,2)
        asV = self.text_model_encoder_layers_4_self_attn_v_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)

        bmmQK = (torch.bmm(input=asQ, mat2=asK).view(k70, k12, k77, k77)+ causal_attention_mask).view(k70x12, k77, k77)
        bmmQK = torch.softmax(input=bmmQK, dim=-1)

        bmmQKV = torch.bmm(input=bmmQK, mat2=asV).view(k70, k12, k77, 64).transpose(1,2).reshape(k70, k77, 768)

        asO = LastO+self.text_model_encoder_layers_4_self_attn_out_proj(bmmQKV)

        Omlp = self.text_model_encoder_layers_4_mlp_fc1(self.text_model_encoder_layers_4_layer_norm2(asO))
        LastO = asO + self.text_model_encoder_layers_4_mlp_fc2((Omlp * torch.sigmoid(input=Omlp * 1.702)))


        On1 = self.text_model_encoder_layers_5_layer_norm1(LastO)

        asQ = (self.text_model_encoder_layers_5_self_attn_q_proj(On1) /8).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)
        asK = self.text_model_encoder_layers_5_self_attn_k_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64).transpose(1,2)
        asV = self.text_model_encoder_layers_5_self_attn_v_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)

        bmmQK = (torch.bmm(input=asQ, mat2=asK).view(k70, k12, k77, k77)+ causal_attention_mask).view(k70x12, k77, k77)
        bmmQK = torch.softmax(input=bmmQK, dim=-1)

        bmmQKV = torch.bmm(input=bmmQK, mat2=asV).view(k70, k12, k77, 64).transpose(1,2).reshape(k70, k77, 768)

        asO = LastO+self.text_model_encoder_layers_5_self_attn_out_proj(bmmQKV)

        Omlp = self.text_model_encoder_layers_5_mlp_fc1(self.text_model_encoder_layers_5_layer_norm2(asO))
        LastO = asO + self.text_model_encoder_layers_5_mlp_fc2((Omlp * torch.sigmoid(input=Omlp * 1.702)))


        On1 = self.text_model_encoder_layers_6_layer_norm1(LastO)

        asQ = (self.text_model_encoder_layers_6_self_attn_q_proj(On1) /8).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)
        asK = self.text_model_encoder_layers_6_self_attn_k_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64).transpose(1,2)
        asV = self.text_model_encoder_layers_6_self_attn_v_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)

        bmmQK = (torch.bmm(input=asQ, mat2=asK).view(k70, k12, k77, k77)+ causal_attention_mask).view(k70x12, k77, k77)
        bmmQK = torch.softmax(input=bmmQK, dim=-1)

        bmmQKV = torch.bmm(input=bmmQK, mat2=asV).view(k70, k12, k77, 64).transpose(1,2).reshape(k70, k77, 768)

        asO = LastO+self.text_model_encoder_layers_6_self_attn_out_proj(bmmQKV)

        Omlp = self.text_model_encoder_layers_6_mlp_fc1(self.text_model_encoder_layers_6_layer_norm2(asO))
        LastO = asO + self.text_model_encoder_layers_6_mlp_fc2((Omlp * torch.sigmoid(input=Omlp * 1.702)))


        On1 = self.text_model_encoder_layers_7_layer_norm1(LastO)

        asQ = (self.text_model_encoder_layers_7_self_attn_q_proj(On1) /8).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)
        asK = self.text_model_encoder_layers_7_self_attn_k_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64).transpose(1,2)
        asV = self.text_model_encoder_layers_7_self_attn_v_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)

        bmmQK = (torch.bmm(input=asQ, mat2=asK).view(k70, k12, k77, k77)+ causal_attention_mask).view(k70x12, k77, k77)
        bmmQK = torch.softmax(input=bmmQK, dim=-1)

        bmmQKV = torch.bmm(input=bmmQK, mat2=asV).view(k70, k12, k77, 64).transpose(1,2).reshape(k70, k77, 768)

        asO = LastO+self.text_model_encoder_layers_7_self_attn_out_proj(bmmQKV)

        Omlp = self.text_model_encoder_layers_7_mlp_fc1(self.text_model_encoder_layers_7_layer_norm2(asO))
        LastO = asO + self.text_model_encoder_layers_7_mlp_fc2((Omlp * torch.sigmoid(input=Omlp * 1.702)))

        if self.cut == -4:
          return LastO

        On1 = self.text_model_encoder_layers_8_layer_norm1(LastO)

        asQ = (self.text_model_encoder_layers_8_self_attn_q_proj(On1) /8).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)
        asK = self.text_model_encoder_layers_8_self_attn_k_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64).transpose(1,2)
        asV = self.text_model_encoder_layers_8_self_attn_v_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)

        bmmQK = (torch.bmm(input=asQ, mat2=asK).view(k70, k12, k77, k77)+ causal_attention_mask).view(k70x12, k77, k77)
        bmmQK = torch.softmax(input=bmmQK, dim=-1)

        bmmQKV = torch.bmm(input=bmmQK, mat2=asV).view(k70, k12, k77, 64).transpose(1,2).reshape(k70, k77, 768)

        asO = LastO+self.text_model_encoder_layers_8_self_attn_out_proj(bmmQKV)

        Omlp = self.text_model_encoder_layers_8_mlp_fc1(self.text_model_encoder_layers_8_layer_norm2(asO))
        LastO = asO + self.text_model_encoder_layers_8_mlp_fc2((Omlp * torch.sigmoid(input=Omlp * 1.702)))

        if self.cut == -3:
          return LastO


        On1 = self.text_model_encoder_layers_9_layer_norm1(LastO)

        asQ = (self.text_model_encoder_layers_9_self_attn_q_proj(On1) /8).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)
        asK = self.text_model_encoder_layers_9_self_attn_k_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64).transpose(1,2)
        asV = self.text_model_encoder_layers_9_self_attn_v_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)

        bmmQK = (torch.bmm(input=asQ, mat2=asK).view(k70, k12, k77, k77)+ causal_attention_mask).view(k70x12, k77, k77)
        bmmQK = torch.softmax(input=bmmQK, dim=-1)

        bmmQKV = torch.bmm(input=bmmQK, mat2=asV).view(k70, k12, k77, 64).transpose(1,2).reshape(k70, k77, 768)

        asO = LastO+self.text_model_encoder_layers_9_self_attn_out_proj(bmmQKV)

        Omlp = self.text_model_encoder_layers_9_mlp_fc1(self.text_model_encoder_layers_9_layer_norm2(asO))
        LastO = asO + self.text_model_encoder_layers_9_mlp_fc2((Omlp * torch.sigmoid(input=Omlp * 1.702)))

        if self.cut == -2:
          return LastO


        On1 = self.text_model_encoder_layers_10_layer_norm1(LastO)

        asQ = (self.text_model_encoder_layers_10_self_attn_q_proj(On1) /8).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)
        asK = self.text_model_encoder_layers_10_self_attn_k_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64).transpose(1,2)
        asV = self.text_model_encoder_layers_10_self_attn_v_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)

        bmmQK = (torch.bmm(input=asQ, mat2=asK).view(k70, k12, k77, k77)+ causal_attention_mask).view(k70x12, k77, k77)
        bmmQK = torch.softmax(input=bmmQK, dim=-1)

        bmmQKV = torch.bmm(input=bmmQK, mat2=asV).view(k70, k12, k77, 64).transpose(1,2).reshape(k70, k77, 768)

        asO = LastO+self.text_model_encoder_layers_10_self_attn_out_proj(bmmQKV)

        Omlp = self.text_model_encoder_layers_10_mlp_fc1(self.text_model_encoder_layers_10_layer_norm2(asO))
        LastO = asO + self.text_model_encoder_layers_10_mlp_fc2((Omlp * torch.sigmoid(input=Omlp * 1.702)))

        if self.cut == -1:
          return LastO

        On1 = self.text_model_encoder_layers_11_layer_norm1(LastO)

        asQ = (self.text_model_encoder_layers_11_self_attn_q_proj(On1) /8).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)
        asK = self.text_model_encoder_layers_11_self_attn_k_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64).transpose(1,2)
        asV = self.text_model_encoder_layers_11_self_attn_v_proj(On1).view(k70, -1, k12, 64).transpose(1,2).reshape(k70x12, -1, 64)

        bmmQK = (torch.bmm(input=asQ, mat2=asK).view(k70, k12, k77, k77)+ causal_attention_mask).view(k70x12, k77, k77)
        bmmQK = torch.softmax(input=bmmQK, dim=-1)

        bmmQKV = torch.bmm(input=bmmQK, mat2=asV).view(k70, k12, k77, 64).transpose(1,2).reshape(k70, k77, 768)

        asO = LastO+self.text_model_encoder_layers_11_self_attn_out_proj(bmmQKV)

        Omlp = self.text_model_encoder_layers_11_mlp_fc1(self.text_model_encoder_layers_11_layer_norm2(asO))
        LastO = asO + self.text_model_encoder_layers_11_mlp_fc2((Omlp * torch.sigmoid(input=Omlp * 1.702)))

        return LastO

from accelerate import init_empty_weights
with init_empty_weights():
    jitbase.transformerJIT= transformer().requires_grad_(False).eval()