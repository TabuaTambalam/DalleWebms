import torch

class config:
    def ckp0(kall,*args):
        return kall(*args)

    def ckp1(kall,*args):
        return torch.utils.checkpoint.checkpoint(kall,*args)

    ckp=ckp0
    pad='zeros'		#'reflect', 'replicate', 'circular'
    mHeadVer=1
