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

        self.time_embed_0 = nn.Linear(bias=True, in_features=160, out_features=640)

        self.time_embed_2 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.input_blocks_0_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=6, kernel_size=(3,3), out_channels=160, padding=(1,1), padding_mode='zeros', stride=(1,1))

        self.input_blocks_1_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=160,eps=0.000010)

        self.input_blocks_1_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=160, kernel_size=(3,3), out_channels=160, padding=(1,1), padding_mode='zeros', stride=(1,1))

        self.input_blocks_1_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=160)
        self.input_blocks_1_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=160,eps=0.000010)

        self.input_blocks_1_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=160, kernel_size=(3,3), out_channels=160, padding=(1,1), padding_mode='zeros', stride=(1,1))

        self.input_blocks_2_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=160,eps=0.000010)

        self.input_blocks_2_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=160, kernel_size=(3,3), out_channels=160, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_2_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=160)
        self.input_blocks_2_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=160,eps=0.000010)

        self.input_blocks_2_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=160, kernel_size=(3,3), out_channels=160, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_3_0_op = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=160, kernel_size=(3,3), out_channels=160, padding=(1,1), padding_mode='zeros', stride=(2,2))

        self.input_blocks_4_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=160,eps=0.000010)

        self.input_blocks_4_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=160, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_4_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=320)
        self.input_blocks_4_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.input_blocks_4_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_4_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=160, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))

        self.input_blocks_5_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.input_blocks_5_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_5_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=320)
        self.input_blocks_5_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.input_blocks_5_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_6_0_op = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(2,2))

        self.input_blocks_7_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.input_blocks_7_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_7_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=320)
        self.input_blocks_7_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.input_blocks_7_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))

        self.input_blocks_8_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.input_blocks_8_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_8_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=320)
        self.input_blocks_8_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.input_blocks_8_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_9_0_op = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(2,2))

        self.input_blocks_10_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.input_blocks_10_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_10_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.input_blocks_10_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.input_blocks_10_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_10_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))

        self.input_blocks_10_1_norm=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)
        self.input_blocks_10_1_qkv = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=640, kernel_size=(1,), out_channels=1920, padding=(0,), padding_mode='zeros', stride=(1,))
        self.input_blocks_10_1_proj_out = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=640, kernel_size=(1,), out_channels=640, padding=(0,), padding_mode='zeros', stride=(1,))

        self.input_blocks_11_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.input_blocks_11_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.input_blocks_11_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.input_blocks_11_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.input_blocks_11_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))

        self.input_blocks_11_1_norm=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)
        self.input_blocks_11_1_qkv = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=640, kernel_size=(1,), out_channels=1920, padding=(0,), padding_mode='zeros', stride=(1,))
        self.input_blocks_11_1_proj_out = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=640, kernel_size=(1,), out_channels=640, padding=(0,), padding_mode='zeros', stride=(1,))

        self.middle_block_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.middle_block_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.middle_block_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.middle_block_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.middle_block_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))

        self.middle_block_1_norm=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)
        self.middle_block_1_qkv = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=640, kernel_size=(1,), out_channels=1920, padding=(0,), padding_mode='zeros', stride=(1,))
        self.middle_block_1_proj_out = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=640, kernel_size=(1,), out_channels=640, padding=(0,), padding_mode='zeros', stride=(1,))

        self.middle_block_2_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.middle_block_2_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.middle_block_2_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.middle_block_2_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.middle_block_2_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))

        self.output_blocks_0_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.output_blocks_0_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_0_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.output_blocks_0_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.output_blocks_0_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_0_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))

        self.output_blocks_0_1_norm=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)
        self.output_blocks_0_1_qkv = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=640, kernel_size=(1,), out_channels=1920, padding=(0,), padding_mode='zeros', stride=(1,))
        self.output_blocks_0_1_proj_out = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=640, kernel_size=(1,), out_channels=640, padding=(0,), padding_mode='zeros', stride=(1,))

        self.output_blocks_1_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=1280,eps=0.000010)

        self.output_blocks_1_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_1_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.output_blocks_1_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.output_blocks_1_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_1_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1280, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))

        self.output_blocks_1_1_norm=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)
        self.output_blocks_1_1_qkv = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=640, kernel_size=(1,), out_channels=1920, padding=(0,), padding_mode='zeros', stride=(1,))
        self.output_blocks_1_1_proj_out = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=640, kernel_size=(1,), out_channels=640, padding=(0,), padding_mode='zeros', stride=(1,))

        self.output_blocks_2_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=960,eps=0.000010)

        self.output_blocks_2_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=960, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_2_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=640)
        self.output_blocks_2_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.output_blocks_2_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_2_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=960, kernel_size=(1,1), out_channels=640, padding=(0,0), padding_mode='zeros', stride=(1,1))

        self.output_blocks_2_1_norm=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)
        self.output_blocks_2_1_qkv = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=640, kernel_size=(1,), out_channels=1920, padding=(0,), padding_mode='zeros', stride=(1,))
        self.output_blocks_2_1_proj_out = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=640, kernel_size=(1,), out_channels=640, padding=(0,), padding_mode='zeros', stride=(1,))
        self.output_blocks_2_2_conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=640, padding=(1,1), padding_mode='zeros', stride=(1,1))

        self.output_blocks_3_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=960,eps=0.000010)

        self.output_blocks_3_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=960, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_3_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=320)
        self.output_blocks_3_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.output_blocks_3_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_3_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=960, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))

        self.output_blocks_4_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.output_blocks_4_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_4_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=320)
        self.output_blocks_4_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.output_blocks_4_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_4_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))

        self.output_blocks_5_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.output_blocks_5_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_5_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=320)
        self.output_blocks_5_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.output_blocks_5_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_5_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_5_1_conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))

        self.output_blocks_6_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.output_blocks_6_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_6_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=320)
        self.output_blocks_6_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.output_blocks_6_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_6_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))

        self.output_blocks_7_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=640,eps=0.000010)

        self.output_blocks_7_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_7_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=320)
        self.output_blocks_7_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.output_blocks_7_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_7_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=640, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))

        self.output_blocks_8_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=480,eps=0.000010)

        self.output_blocks_8_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=480, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_8_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=320)
        self.output_blocks_8_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.output_blocks_8_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_8_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=480, kernel_size=(1,1), out_channels=320, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.output_blocks_8_1_conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))

        self.output_blocks_9_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=480,eps=0.000010)

        self.output_blocks_9_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=480, kernel_size=(3,3), out_channels=160, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_9_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=160)
        self.output_blocks_9_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=160,eps=0.000010)

        self.output_blocks_9_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=160, kernel_size=(3,3), out_channels=160, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_9_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=480, kernel_size=(1,1), out_channels=160, padding=(0,0), padding_mode='zeros', stride=(1,1))

        self.output_blocks_10_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.output_blocks_10_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=160, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_10_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=160)
        self.output_blocks_10_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=160,eps=0.000010)

        self.output_blocks_10_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=160, kernel_size=(3,3), out_channels=160, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_10_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(1,1), out_channels=160, padding=(0,0), padding_mode='zeros', stride=(1,1))

        self.output_blocks_11_0_in_layers_0=nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)

        self.output_blocks_11_0_in_layers_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=160, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_11_0_emb_layers_1 = nn.Linear(bias=True, in_features=640, out_features=160)
        self.output_blocks_11_0_out_layers_0=nn.GroupNorm(num_groups=32,num_channels=160,eps=0.000010)

        self.output_blocks_11_0_out_layers_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=160, kernel_size=(3,3), out_channels=160, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.output_blocks_11_0_skip_connection = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(1,1), out_channels=160, padding=(0,0), padding_mode='zeros', stride=(1,1))

        self.out_0=nn.GroupNorm(num_groups=32,num_channels=160,eps=0.000010)

        self.out_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=160, kernel_size=(3,3), out_channels=3, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.freqs=nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, v_0, v_1):

        v_106 = v_1.unsqueeze(1) * self.freqs
        v_117 = F.silu(self.time_embed_2(F.silu(self.time_embed_0(torch.cat((torch.cos(v_106), torch.sin(v_106)), dim=-1)))))

        v_113 = self.input_blocks_0_0(v_0)

        v_125 = v_113 + self.input_blocks_1_0_out_layers_3(F.silu(self.input_blocks_1_0_out_layers_0(self.input_blocks_1_0_in_layers_2(F.silu(self.input_blocks_1_0_in_layers_0(v_113))) + self.input_blocks_1_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))

        v_136 = v_125 + self.input_blocks_2_0_out_layers_3(F.silu(self.input_blocks_2_0_out_layers_0(self.input_blocks_2_0_in_layers_2(F.silu(self.input_blocks_2_0_in_layers_0(v_125))) + self.input_blocks_2_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))
        v_137 = self.input_blocks_3_0_op(v_136)

        v_149 = self.input_blocks_4_0_skip_connection(v_137) + self.input_blocks_4_0_out_layers_3(F.silu(self.input_blocks_4_0_out_layers_0(self.input_blocks_4_0_in_layers_2(F.silu(self.input_blocks_4_0_in_layers_0(v_137))) + self.input_blocks_4_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))

        v_160 = v_149 + self.input_blocks_5_0_out_layers_3(F.silu(self.input_blocks_5_0_out_layers_0(self.input_blocks_5_0_in_layers_2(F.silu(self.input_blocks_5_0_in_layers_0(v_149))) + self.input_blocks_5_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))
        v_161 = self.input_blocks_6_0_op(v_160)

        v_172 = v_161 + self.input_blocks_7_0_out_layers_3(F.silu(self.input_blocks_7_0_out_layers_0(self.input_blocks_7_0_in_layers_2(F.silu(self.input_blocks_7_0_in_layers_0(v_161))) + self.input_blocks_7_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))

        v_183 = v_172 + self.input_blocks_8_0_out_layers_3(F.silu(self.input_blocks_8_0_out_layers_0(self.input_blocks_8_0_in_layers_2(F.silu(self.input_blocks_8_0_in_layers_0(v_172))) + self.input_blocks_8_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))
        v_184 = self.input_blocks_9_0_op(v_183)

        v_197 = (self.input_blocks_10_0_skip_connection(v_184) + self.input_blocks_10_0_out_layers_3(F.silu(self.input_blocks_10_0_out_layers_0(self.input_blocks_10_0_in_layers_2(F.silu(self.input_blocks_10_0_in_layers_0(v_184))) + self.input_blocks_10_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))).reshape(1, 640, -1)
        v_201, v_202, v_203 = torch.split(tensor=self.input_blocks_10_1_qkv(self.input_blocks_10_1_norm(v_197)).reshape(20, 96, -1), dim=1, split_size_or_sections=32)
        v_209 = torch.einsum('ikl,ijl->ijk',F.softmax(input=torch.einsum('ilj,ilk->ijk', v_201, v_202)* 1.767767e-01, dim=-1), v_203).reshape(1, -1, 256)
        v_212 = (v_197 + self.input_blocks_10_1_proj_out(v_209)).reshape(1, 640, 16, 16)

        v_224 = (v_212 + self.input_blocks_11_0_out_layers_3(F.silu(self.input_blocks_11_0_out_layers_0(self.input_blocks_11_0_in_layers_2(F.silu(self.input_blocks_11_0_in_layers_0(v_212))) + self.input_blocks_11_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))).reshape(1, 640, -1)
        v_228, v_229, v_230 = torch.split(tensor=self.input_blocks_11_1_qkv(self.input_blocks_11_1_norm(v_224)).reshape(20, 96, -1), dim=1, split_size_or_sections=32)
        v_236 = torch.einsum('ikl,ijl->ijk', F.softmax(input=torch.einsum('ilj,ilk->ijk', v_228 , v_229)* 1.767767e-01, dim=-1), v_230).reshape(1, -1, 256)
        v_239 = (v_224 + self.input_blocks_11_1_proj_out(v_236)).reshape(1, 640, 16, 16)

        v_251 = (v_239 + self.middle_block_0_out_layers_3(F.silu(self.middle_block_0_out_layers_0(self.middle_block_0_in_layers_2(F.silu(self.middle_block_0_in_layers_0(v_239))) + self.middle_block_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))).reshape(1, 640, -1)
        v_255, v_256, v_257 = torch.split(tensor=self.middle_block_1_qkv(self.middle_block_1_norm(v_251)).reshape(20, 96, -1), dim=1, split_size_or_sections=32)
        v_263 = torch.einsum('ikl,ijl->ijk', F.softmax(input=torch.einsum('ilj,ilk->ijk', v_255 , v_256)* 1.767767e-01, dim=-1), v_257).reshape(1, -1, 256)
        v_266 = (v_251 + self.middle_block_1_proj_out(v_263)).reshape(1, 640, 16, 16)

        v_278 = torch.cat(((v_266 + self.middle_block_2_out_layers_3(F.silu(self.middle_block_2_out_layers_0(self.middle_block_2_in_layers_2(F.silu(self.middle_block_2_in_layers_0(v_266))) + self.middle_block_2_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))), v_239), dim=1)

        v_291 = (self.output_blocks_0_0_skip_connection(v_278) + self.output_blocks_0_0_out_layers_3(F.silu(self.output_blocks_0_0_out_layers_0(self.output_blocks_0_0_in_layers_2(F.silu(self.output_blocks_0_0_in_layers_0(v_278))) + self.output_blocks_0_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))).reshape(1, 640, -1)
        v_295, v_296, v_297 = torch.split(tensor=self.output_blocks_0_1_qkv(self.output_blocks_0_1_norm(v_291)).reshape(20, 96, -1), dim=1, split_size_or_sections=32)
        v_303 = torch.einsum('ikl,ijl->ijk', F.softmax(input=torch.einsum('ilj,ilk->ijk', v_295 , v_296 )* 1.767767e-01, dim=-1), v_297).reshape(1, -1, 256)
        v_307 = torch.cat(((v_291 + self.output_blocks_0_1_proj_out(v_303)).reshape(1, 640, 16, 16), v_212), dim=1)

        v_320 = (self.output_blocks_1_0_skip_connection(v_307) + self.output_blocks_1_0_out_layers_3(F.silu(self.output_blocks_1_0_out_layers_0(self.output_blocks_1_0_in_layers_2(F.silu(self.output_blocks_1_0_in_layers_0(v_307))) + self.output_blocks_1_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))).reshape(1, 640, -1)
        v_324, v_325, v_326 = torch.split(tensor=self.output_blocks_1_1_qkv(self.output_blocks_1_1_norm(v_320)).reshape(20, 96, -1), dim=1, split_size_or_sections=32)
        v_331 = torch.einsum('ikl,ijl->ijk', F.softmax(input=torch.einsum('ilj,ilk->ijk', v_324 , v_325)* 1.767767e-01, dim=-1), v_326).reshape(1, -1, 256)
        v_336 = torch.cat(((v_320 + self.output_blocks_1_1_proj_out(v_331)).reshape(1, 640, 16, 16), v_184), dim=1)

        v_349 = (self.output_blocks_2_0_skip_connection(v_336) + self.output_blocks_2_0_out_layers_3(F.silu(self.output_blocks_2_0_out_layers_0(self.output_blocks_2_0_in_layers_2(F.silu(self.output_blocks_2_0_in_layers_0(v_336))) + self.output_blocks_2_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))).reshape(1, 640, -1)
        v_353, v_354, v_355 = torch.split(tensor=self.output_blocks_2_1_qkv(self.output_blocks_2_1_norm(v_349)).reshape(20, 96, -1), dim=1, split_size_or_sections=32)
        v_360 = torch.einsum('ikl,ijl->ijk', F.softmax(input=torch.einsum('ilj,ilk->ijk', v_353 , v_354 )* 1.767767e-01, dim=-1), v_355).reshape(1, -1, 256)
        v_364 = (v_349 + self.output_blocks_2_1_proj_out(v_360)).reshape(1, 640, 16, 16)

        v_367 = torch.cat((self.output_blocks_2_2_conv(F.interpolate(v_364, scale_factor=2.0,mode='nearest')), v_183), dim=1)

        v_380 = torch.cat(((self.output_blocks_3_0_skip_connection(v_367) + self.output_blocks_3_0_out_layers_3(F.silu(self.output_blocks_3_0_out_layers_0(self.output_blocks_3_0_in_layers_2(F.silu(self.output_blocks_3_0_in_layers_0(v_367))) + self.output_blocks_3_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))), v_172), dim=1)

        v_393 = torch.cat(((self.output_blocks_4_0_skip_connection(v_380) + self.output_blocks_4_0_out_layers_3(F.silu(self.output_blocks_4_0_out_layers_0(self.output_blocks_4_0_in_layers_2(F.silu(self.output_blocks_4_0_in_layers_0(v_380))) + self.output_blocks_4_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))), v_161), dim=1)

        v_405 = self.output_blocks_5_0_skip_connection(v_393) + self.output_blocks_5_0_out_layers_3(F.silu(self.output_blocks_5_0_out_layers_0(self.output_blocks_5_0_in_layers_2(F.silu(self.output_blocks_5_0_in_layers_0(v_393))) + self.output_blocks_5_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))
        v_408 = torch.cat((self.output_blocks_5_1_conv(F.interpolate(v_405, scale_factor=2.0,mode='nearest')), v_160), dim=1)

        v_421 = torch.cat(((self.output_blocks_6_0_skip_connection(v_408) + self.output_blocks_6_0_out_layers_3(F.silu(self.output_blocks_6_0_out_layers_0(self.output_blocks_6_0_in_layers_2(F.silu(self.output_blocks_6_0_in_layers_0(v_408))) + self.output_blocks_6_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))), v_149), dim=1)

        v_434 = torch.cat(((self.output_blocks_7_0_skip_connection(v_421) + self.output_blocks_7_0_out_layers_3(F.silu(self.output_blocks_7_0_out_layers_0(self.output_blocks_7_0_in_layers_2(F.silu(self.output_blocks_7_0_in_layers_0(v_421))) + self.output_blocks_7_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))), v_137), dim=1)

        v_446 = self.output_blocks_8_0_skip_connection(v_434) + self.output_blocks_8_0_out_layers_3(F.silu(self.output_blocks_8_0_out_layers_0(self.output_blocks_8_0_in_layers_2(F.silu(self.output_blocks_8_0_in_layers_0(v_434))) + self.output_blocks_8_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))
        v_449 = torch.cat((self.output_blocks_8_1_conv(F.interpolate(v_446, scale_factor=2.0,mode='nearest')), v_136), dim=1)


        v_462 = torch.cat(((self.output_blocks_9_0_skip_connection(v_449) + self.output_blocks_9_0_out_layers_3(F.silu(self.output_blocks_9_0_out_layers_0(self.output_blocks_9_0_in_layers_2(F.silu(self.output_blocks_9_0_in_layers_0(v_449))) + self.output_blocks_9_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))), v_125), dim=1)

        v_475 = torch.cat(((self.output_blocks_10_0_skip_connection(v_462) + self.output_blocks_10_0_out_layers_3(F.silu(self.output_blocks_10_0_out_layers_0(self.output_blocks_10_0_in_layers_2(F.silu(self.output_blocks_10_0_in_layers_0(v_462))) + self.output_blocks_10_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))), v_113), dim=1)

  
        v_487 = self.output_blocks_11_0_skip_connection(v_475) + self.output_blocks_11_0_out_layers_3(F.silu(self.output_blocks_11_0_out_layers_0(self.output_blocks_11_0_in_layers_2(F.silu(self.output_blocks_11_0_in_layers_0(v_475))) + self.output_blocks_11_0_emb_layers_1(v_117).unsqueeze(2).unsqueeze(3))))

        return self.out_2(F.silu(self.out_0(v_487)))

