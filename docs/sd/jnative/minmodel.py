import torch
import torch.nn as nn
import torch.nn.functional as F

from jitbase import config
from unet_blocks import ResnetBlock, AttentionBlock, AttentionBlock_conv


def callRA(mdlist,x,emb,cond_k,cond_v):
    m0=config.ckp(mdlist[0].forward,x,emb)
    return config.ckp(mdlist[1].forward,m0, cond_k,cond_v)


def callRC(mdlist,x,emb):
    m0 = config.ckp(mdlist[0].forward,x,emb)
    return config.ckp(mdlist[1].forward,m0)

def callRAC(mdlist,x,emb,cond_k,cond_v):
    m0 = config.ckp(mdlist[0].forward,x,emb)
    m1 = config.ckp(mdlist[1].forward,m0, cond_k,cond_v)
    return config.ckp(mdlist[2].forward,m1)



class UNetModel(nn.Module):

    def __init__(
        self,
        in_channels = 4
    ):
        super().__init__()


        # input
        self.input_blocks_0_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=in_channels, kernel_size=(3,3), out_channels=320, padding=(1,1), padding_mode='zeros', stride=(1,1))
        # time
        self.freqs=nn.Parameter(torch.ones(1), requires_grad=False)
        self.time_embed_0 = nn.Linear(bias=True, in_features=320, out_features=1280)
        self.time_embed_2 = nn.Linear(bias=True, in_features=1280, out_features=1280)

        self.input_blocks = nn.ModuleList([None]*12)
        self.input_blocks[1]=nn.ModuleList([ResnetBlock(320,320),AttentionBlock(320,1280)])
        self.input_blocks[2]=nn.ModuleList([ResnetBlock(320,320),AttentionBlock(320,1280)])

        self.input_blocks[3]=nn.ModuleList([AttentionBlock_conv(320,2)])
        self.input_blocks[4]=nn.ModuleList([ResnetBlock(320,640,prv_skip=True),AttentionBlock(640,2560)])
        self.input_blocks[5]=nn.ModuleList([ResnetBlock(640,640),AttentionBlock(640,2560)])

        self.input_blocks[6]=nn.ModuleList([AttentionBlock_conv(640,2)])
        self.input_blocks[7]=nn.ModuleList([ResnetBlock(640,1280,prv_skip=True),AttentionBlock(1280,5120)])	#9
        self.input_blocks[8]=nn.ModuleList([ResnetBlock(1280,1280),AttentionBlock(1280,5120)])		#11

        self.input_blocks[9]=nn.ModuleList([AttentionBlock_conv(1280,2)])
        self.input_blocks[10]=nn.ModuleList([ResnetBlock(1280,1280)])
        self.input_blocks[11]=nn.ModuleList([ResnetBlock(1280,1280)])

        self.middle_block = nn.ModuleList([ResnetBlock(1280,1280),AttentionBlock(1280,5120),ResnetBlock(1280,1280)])	#13

        self.output_blocks = nn.ModuleList([None]*12)
        self.output_blocks[0]=nn.ModuleList([ResnetBlock(2560,1280,prv_skip=True)])
        self.output_blocks[1]=nn.ModuleList([ResnetBlock(2560,1280,prv_skip=True)])
        self.output_blocks[2]=nn.ModuleList([ResnetBlock(2560,1280,prv_skip=True),		AttentionBlock_conv(1280,1)])

        self.output_blocks[3]=nn.ModuleList([ResnetBlock(2560,1280,prv_skip=True),AttentionBlock(1280,5120,cat_prev=True)])	#15
        self.output_blocks[4]=nn.ModuleList([ResnetBlock(2560,1280,prv_skip=True),AttentionBlock(1280,5120,cat_prev=True)])	#17
        self.output_blocks[5]=nn.ModuleList([ResnetBlock(1920,1280,prv_skip=True),AttentionBlock(1280,5120),AttentionBlock_conv(1280,1)])	#19

        self.output_blocks[6]=nn.ModuleList([ResnetBlock(1920,640,prv_skip=True),AttentionBlock(640,2560,cat_prev=True)])
        self.output_blocks[7]=nn.ModuleList([ResnetBlock(1280,640,prv_skip=True),AttentionBlock(640,2560,cat_prev=True)])
        self.output_blocks[8]=nn.ModuleList([ResnetBlock(960, 640,prv_skip=True),AttentionBlock(640,2560),AttentionBlock_conv(640,1)])

        self.output_blocks[9]=nn.ModuleList([ResnetBlock(960,320,prv_skip=True),AttentionBlock(320,1280,cat_prev=True)])
        self.output_blocks[10]=nn.ModuleList([ResnetBlock(640,320,prv_skip=True),AttentionBlock(320,1280,cat_prev=True)])
        self.output_blocks[11]=nn.ModuleList([ResnetBlock(640,320,prv_skip=True),AttentionBlock(320,1280)])

        # out
        self.out_0 = nn.GroupNorm(num_groups=32,num_channels=320,eps=0.000010)
        self.out_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=320, kernel_size=(3,3), out_channels=4, padding=(1,1), padding_mode='zeros', stride=(1,1))

    def time_embedding(
        self,
        t
    ):
        v_5 = t.unsqueeze(1) * self.freqs
        return F.silu(self.time_embed_2(F.silu(self.time_embed_0(torch.cat((torch.cos(v_5), torch.sin(v_5)), dim=-1)))))
        


    def forward_crossattn(
        self,
        x,
        t,
        context_k,
        context_v=None
    ):
        if context_v is None:
          context_v=context_k
        # 1. time
        emb = self.time_embedding(t)
        hs=[None]*12

        h0 = self.input_blocks_0_0(x)
        hs[0]=h0
        h1=callRA(self.input_blocks[1],h0,emb,context_k,context_v)
        hs[1]=h1
        h2=callRA(self.input_blocks[2],h1,emb,context_k,context_v)
        hs[2]=h2
        h3=config.ckp(self.input_blocks[3][0].forward,h2)
        hs[3]=h3
        h4=callRA(self.input_blocks[4],h3,emb,context_k,context_v)
        hs[4]=h4
        h5=callRA(self.input_blocks[5],h4,emb,context_k,context_v)
        hs[5]=h5
        h6=config.ckp(self.input_blocks[6][0].forward,h5)
        hs[6]=h6
        h7=callRA(self.input_blocks[7],h6,emb,context_k,context_v)
        hs[7]=h7
        h8=callRA(self.input_blocks[8],h7,emb,context_k,context_v)
        hs[8]=h8
        h9=config.ckp(self.input_blocks[9][0].forward,h8)
        hs[9]=h9
        h10=config.ckp(self.input_blocks[10][0].forward,h9,emb)
        hs[10]=h10
        h11=config.ckp(self.input_blocks[11][0].forward,h10,emb)
        hs[11]=h11

        h=config.ckp(self.middle_block[0].forward,h11,emb)
        h=config.ckp(self.middle_block[1].forward,h,context_k,context_v)
        h=config.ckp(self.middle_block[2].forward,h,emb)
        return h,emb,hs

    def forward2(self, h, emb, context_k, h6, h7, h8, h9, h10, h11,context_v=None):
        if context_v is None:
          context_v=context_k
        h0 = torch.cat((h, h11), dim=1)
        h0 = torch.cat((config.ckp(self.output_blocks[0][0].forward,h0,emb), h10), dim=1)
        h0 = torch.cat((config.ckp(self.output_blocks[1][0].forward,h0,emb), h9), dim=1)
        h0 = torch.cat((callRC(self.output_blocks[2],h0,emb), h8), dim=1)
        h0 = torch.cat((callRA(self.output_blocks[3],h0,emb,context_k,context_v), h7), dim=1)
        h0 = torch.cat((callRA(self.output_blocks[4],h0,emb,context_k,context_v), h6), dim=1)
        h0 = callRAC(self.output_blocks[5],h0,emb,context_k,context_v)
        return h0

    def forward3(self, h, emb, context_k, h0, h1, h2, h3, h4, h5,context_v=None):
        if context_v is None:
          context_v=context_k
        hv = torch.cat((h, h5), dim=1)
        hv = torch.cat((callRA(self.output_blocks[6],hv,emb,context_k,context_v), h4), dim=1)
        hv = torch.cat((callRA(self.output_blocks[7],hv,emb,context_k,context_v), h3), dim=1)
        hv = torch.cat((callRAC(self.output_blocks[8],hv,emb,context_k,context_v), h2), dim=1)
        hv = torch.cat((callRA(self.output_blocks[9],hv,emb,context_k,context_v), h1), dim=1)
        hv = torch.cat((callRA(self.output_blocks[10],hv,emb,context_k,context_v), h0), dim=1)
        hv = callRA(self.output_blocks[11],hv,emb,context_k,context_v)
        return self.out_2(F.silu(self.out_0(hv)))