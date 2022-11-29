import os
import torch
import hashlib
import numpy as np
torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True



def get_keys_to_submodule(model):
    keys_to_submodule = {}
    # iterate all submodules
    for submodule_name, submodule in model.named_modules():
        # iterate all paramters in each submobule
        for param_name, param in submodule.named_parameters():
            # param_name is organized as <name>.<subname>.<subsubname> ...
            splitted_param_name = param_name.split('.')
            # we cannot go inside it anymore. This is the actual parameter
            is_leaf_param = len(splitted_param_name) == 1
            if is_leaf_param:
                # we recreate the correct key
                key = f"{submodule_name}.{param_name}"
                # we associate this key with this submodule
                keys_to_submodule[key] = submodule
                
    return keys_to_submodule

def load_state_dict_with_low_memory(model, state_dict):
    print('======hacky load======')
    keys_to_submodule = get_keys_to_submodule(model)
    mste=model.state_dict()
    for key, submodule in keys_to_submodule.items():
        k2='cond_stage_model.'+key
        # get the valye from the state_dict
        if k2 in state_dict:
          val = state_dict[k2]
        else:
          print(key)
          val = mste[key]

        param_name = key.split('.')[-1]
        new_val = torch.nn.Parameter(val,requires_grad=False)
        setattr(submodule, param_name, new_val)


def load_state_dict_with_low_memoryRe(model, state_dict):
    print('======hacky load======')
    keys_to_submodule = get_keys_to_submodule(model)
    mste=model.state_dict()
    for key, submodule in keys_to_submodule.items():
        # get the valye from the state_dict
        if key.startswith('.'):
          key=key[1:]
        if key in state_dict:
          val = state_dict[key]
        else:
          print(key)
          val = mste[key]

        param_name = key.split('.')[-1]
        new_val = torch.nn.Parameter(val,requires_grad=False)
        setattr(submodule, param_name, new_val)

        
def load_state_dict_with_low_memory_meta(model):
    print('======2meta======')
    metadev=torch.device('meta')
    keys_to_submodule = get_keys_to_submodule(model)
    mste=model.state_dict()
    for key, submodule in keys_to_submodule.items():
        if key.startswith('.'):
          key=key[1:]
        val = mste[key].to(metadev)

        param_name = key.split('.')[-1]
        new_val = torch.nn.Parameter(val,requires_grad=False)
        setattr(submodule, param_name, new_val)

def hashweight(sd):
  kizu=dict()
  for k in sd:
    try:
      kizu[hashlib.md5(sd[k].half().numpy().tobytes()).hexdigest()]=k
    except:
      print(k)
      print(sd[k])
  return kizu

def gprint(sub,base):
  for k in sub:
    if k in base:
      print("'"+sub[k]+"':'"+base[k]+"',")
    else:
      print('# '+sub[k])





ldmpfx='model.diffusion_model.'

def conv2linr(db,k):
  prm=db[ldmpfx+k]
  if '.proj_' in k and '.weight' in k:
    sz=list(prm.shape)
    prm=prm.reshape(sz[:2])
  return prm

def mkmodel_state_dict(zdk,difjit=None):
  import jkt
  def kfeed(pfx,k,k_dst,apnd):
    ret = k+apnd
    jna1[k_dst+apnd]=ret
    return pfx+ret

  model_state_dict_colect=[]
  if difjit is None:
    difjit=[diffusion_emb,diffusion_mid,diffusion_out]
  
  jna1=jkt.nam1
  
  for i in range(3):
    model_state_dict = {}
    sd=difjit[i].state_dict()
    modkeys=jkt.modkeys4[i]
    modkeys_flat=jkt.modkeys3[i]
    mdk_l=len(modkeys)
    for i in range(mdk_l):
      k=modkeys[i]
      ldmpfx_k=ldmpfx+k
      qkv_at1=[zdk[ldmpfx_k+'1.to_q.weight'],
          zdk[ldmpfx_k+'1.to_k.weight'],
          zdk[ldmpfx_k+'1.to_v.weight']]
      wsz=qkv_at1[0].size(0)*3
      mdf=modkeys_flat[i]
      zdk[kfeed(ldmpfx,k,mdf,'1.in_proj_weight')]=torch.cat(qkv_at1)
      zdk[kfeed(ldmpfx,k,mdf,'1.in_proj_bias')] =torch.zeros(wsz)
      zdk[kfeed(ldmpfx,k,mdf,'2.in_proj_bias')] =torch.zeros(wsz)



    for k in sd:
      if k == 'freqs':
        model_state_dict[k]=torch.tensor(np.load('freqs.npy'))
      else:
        model_state_dict[k]=conv2linr(zdk,jna1[k])
    model_state_dict_colect.append(model_state_dict)
  return model_state_dict_colect

if __name__ == '__main__':
  diffusion_emb = torch.jit.load('web/diffusion_emb_pnnx.pt')
  diffusion_mid = torch.jit.load('web/diffusion_mid_pnnx.pt')
  diffusion_out = torch.jit.load('web/diffusion_out_pnnx.pt')
  kek=torch.load('tojit.ckpt',map_location=torch.device('cpu'))
  dkole=mkmodel_state_dict(kek['state_dict'])
  load_state_dict_with_low_memoryRe(diffusion_emb,dkole[0])
  load_state_dict_with_low_memoryRe(diffusion_mid,dkole[1])
  load_state_dict_with_low_memoryRe(diffusion_out,dkole[2])
  with torch.jit.optimized_execution(True):
    diffusion_emb.save('ckpts/diffusion_emb_pnnx.pt')
    diffusion_mid.save('ckpts/diffusion_mid_pnnx.pt')
    diffusion_out.save('ckpts/diffusion_out_pnnx.pt')
