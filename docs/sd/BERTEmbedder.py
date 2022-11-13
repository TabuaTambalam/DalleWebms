import os
import numpy as np
import torch
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
max_length = 77
dedup=dict()
transformer = torch.jit.load('transformer_pnnx.pt')
embedding = torch.nn.Embedding.from_pretrained(transformer.text_model_embeddings_token_embedding.weight)
n_samples=1
t_enc=50
Max_EMB_len=0

def toker(text, pad=False):
  padstr='do_not_pad'
  if pad:
    padstr='max_length'
  batch_encoding = tokenizer(text, truncation=True, max_length=max_length, return_length=True,
                        return_overflowing_tokens=False, padding=padstr, return_tensors='pt')

  return batch_encoding['input_ids'][0]

emp=toker('')
tok_bos, tok_eos = int(emp[0]), int(emp[1])
emp=embedding(emp)
emb_bos, emb_eos = emp[0] ,emp[1]

def encode(text):
  tokenz=toker(text)
  l_tok=tokenz.size(0)
  poz_dict=[None]*l_tok
  for i in range(l_tok):
    toki=int(tokenz[i])
    poz_dict[i]=(tokenizer.decode(toki),toki )
  emp=torch.cat(( embedding(tokenz),emb_eos.expand(max_length-l_tok,-1) )).unsqueeze(0)

  return transformer(emp)[:,:l_tok,:],poz_dict

def encode_complex(text):
  c=makeCs(text)[0]
  tokenz=[]
  for txt,tk in c.txt:
    tokenz.append(tk)
  tokenz=torch.cat(tokenz)
  l_tok=tokenz.size(0)
  poz_dict=[None]*(l_tok+2)
  poz_dict[0]=('BOS',tok_bos)
  for i in range(l_tok):
    toki=int(tokenz[i])
    if toki == -1:
      poz_dict[i+1]=('-YourEmb-',-1 )
    else:
      poz_dict[i+1]=(tokenizer.decode(toki),toki )
  poz_dict[-1]=('EOS',tok_eos)

  return c.arr, poz_dict

def insert(inz):
  dedup[inz]=torch.tensor(np.fromfile('UserEmb/'+inz[1:-1]+'.bin',dtype=np.float32).reshape(-1,768))

def mk_emb_wgt(unit_arr,dtal=-1):
  if dtal < 0:
    dtal=len(unit_arr)

  emb_cur=[]
  emb=[emb_cur]
  count=max_length-2
  wgt=[None]*(dtal+2)
  txt=[]
  
  wgt[0]=torch.ones(1)
  
  NoWgt=True
  for i in range(dtal):
    emb0,wgt0,tok0=unit_arr[i].emb_wgt()
    if wgt0[0] != 1.0:
      NoWgt=False
    msg = unit_arr[i].msg
    if len(msg) > 1:
      txt.append((msg,tok0))
    else:
      txt.append(('--emb--',tok0))
    emb_cur.append(emb0)
    wgt[i+1]=wgt0
    count-=wgt0.size(0)
    if count == 0:
      count=max_length-2
      emb_cur=[]
      emb.append(emb_cur)
    elif count < 0:
      wgt0size=wgt0.size(0)
      if wgt0size < 76:
        del emb_cur[-1]
        count=max_length-2-wgt0size
        emb_cur=[emb0]
        emb.append(emb_cur)
      else:
        del emb_cur[-1]
        count+=wgt0size
  
  if emb_cur:
    count+=1
  else:
    del emb[-1]
    count=1

  emb_l=len(emb)

  for i in range(emb_l):
    emb[i]=torch.cat(emb[i])

  if NoWgt:
    wgt=None
  else:
    wgt[-1]=wgt[0].expand(count)


  if not txt:
    txt=None
  return emb, wgt, txt

def makez0(emb0):
  global Max_EMB_len
  global eblsum
  eblsum=0
  emb_1=len(emb0)
  zout=[None]*(emb_1+2)
  
  max_i_end=0
  zout[-1]=torch.zeros(77,768)
  for i in range(emb_1):
    eee=emb0[i]
    esz=eee.size(0)
    eblsum+=esz
    zuu=emb_eos.expand(max_length,-1).clone()
    zuu[0]=emb_bos
    zuu[1:esz+1]=eee
    zuu=transformer( zuu.unsqueeze(0) )[0]
    zout[0]=zuu[0].unsqueeze(0)
    zout[i+1]=zuu[1:esz+1]
    lyy=zuu[esz+1:]
    i_end=lyy.size(0)+i
    if i_end > max_length:
      lyy=lyy[:max_length-i_end]
      i_end=max_length
    if i_end > max_i_end:
      max_i_end=i_end
    zout[-1][i:i_end]=lyy
  zout[-1]=zout[-1][:max_i_end]
    
  ret=torch.cat(zout).unsqueeze(0)
  curMax_EMB_len=ret.size(1)
  if curMax_EMB_len > Max_EMB_len:
    Max_EMB_len=curMax_EMB_len
  return ret, max_i_end
    

def from_emb(emb0,wgt_arr=None,nsamp=1,cuda=True):
  z, max_i_end = makez0(emb0)

  if wgt_arr is not None:
    wgt_arr[-1]=torch.ones(max_i_end)
    wgt=torch.cat(wgt_arr)

    ynt=z[:,0,:]
    wgt /= torch.abs(wgt.mean())
    z*=wgt.reshape(-1,1).expand(1,-1,-1)
    z[:,0,:]=ynt

  if nsamp > 1:
    z=z.expand(nsamp,-1,-1)
  return z


import copy
import random


class vinfo:
  def __init__(self, txt):
    self.tag=txt
    valid=False
    cut=-1
    if txt[-3] == ',':
      cut=-3
    key=txt[:cut]
    if key in dedup:
      valid=True
      self.bazkey=key
      ActivedPromptVars[txt]=dedup[key]
    self.valid=valid



  @property
  def baz(self):
    return dedup[self.bazkey]

  def repl(self, unit, v):
    inzt=copy.deepcopy(self.baz.varias[v])
    idset=self.baz.ids[v]
    _, p_wgt, p_sta, p_end = unit.nfo()

    idmap=dict()
    procid=False
    if idset:
      procid=True
      for id in idset:
        idmap[id]=rdmIDfunc(None)

    if procid:
      for yn in inzt:
        if yn.id == 0:
          yn.update_sta_end(p_wgt,p_sta,p_end)
        else:
          yn.id=idmap[yn.id]
          erz=yn.eraz
          if erz:
            nyu_erz=dict()
            for k in erz:
              nyu_id=idmap[k]
              nyu_erz[nyu_id]=nyu_id
            yn.eraz=nyu_erz
    else:
      for yn in inzt:
        yn.update_sta_end(p_wgt,p_sta,p_end)

       
    return inzt





class sentUnit:
  def __init__(self, txt,
               fast=-1,
               p_wgt=None,p_sta=None,p_end=None
               ):

    self.emb_wgt = self.emb_wgt0
    self.id=0
    usig=0
    self.repls=dict()
    self.eraz=None
    if fast == 0:
      self.sig=usig
      self.msg, self.wgt, self.upper, self.lower =txt,p_wgt,p_sta,p_end
      return
    elif fast == 1:
      self.id=rdmIDfunc(txt) #id(self)#
      self.sig=usig
      self.msg, self.wgt, self.upper, self.lower =txt,0,p_sta,p_end
      #wgt as group_len
      self.eraz=dict()
      self.yetproc=True
      return

    
    
    prepand=None
    retThis=True
    self.real_return=[]
    if ':' in txt:
      usig+=0x100
    if '+' in txt:
      usig+=0x200
    if ';' in txt:
      usig+=0x400
    if '|' in txt:
      usig+=0x800
    
    self.sig=usig

    if usig < 0x100:
      self.msg, self.wgt, self.upper, self.lower = txt,p_wgt,p_sta,p_end
      self.real_return=[self]
    else:
      taps=mktaps(txt,  p_wgt=p_wgt,p_sta=p_sta,p_end=p_end)
      msg, self.wgt, self.upper, self.lower=taps[0]
      if '|' in msg:
        self.msg='^'
        retThis=False
        taps2=mktaps(msg, sep='|', p_wgt=self.wgt,p_sta=self.upper,p_end=self.lower)
        for m2,w2,s2,e2 in taps2:
          dmm=sentUnit(m2,p_wgt=w2,p_sta=s2,p_end=e2)
          for dmm2 in get_real_return(dmm):
            self.real_return.append(dmm2)
      else:
        self.msg=msg
        self.real_return=[self]
      
      len_taps=len(taps)
      
      if len_taps > 1:
        prepand=sentUnit('[',fast=1, p_sta=self.upper,p_end=self.lower )
        lazt_id=prepand.id
        eraz_arr=[prepand]
        for i in range(1,len_taps):
          msg,wgt_bs,sta_bs,endo_bs = taps[i]
          edb=sentUnit(']',fast=1)
          edb.id=lazt_id
          pp2=sentUnit('[',fast=1,p_sta=sta_bs,p_end=endo_bs)
          lazt_id=pp2.id
          if '|' in msg:
            pral=[edb,pp2]
            taps2=mktaps(msg, sep='|', p_wgt=wgt_bs,p_sta=sta_bs,p_end=endo_bs)
            for m2,w2,s2,e2 in taps2:
              dmm=sentUnit(m2,p_wgt=w2,p_sta=s2,p_end=e2)
              for dmm2 in get_real_return(dmm):
                pral.append(dmm2) 
            self.real_return+=pral

          else:
            dmm=sentUnit(msg,p_wgt=wgt_bs, p_sta=sta_bs,p_end=endo_bs)
            dmm_rt=get_real_return(dmm)
            self.real_return+=[edb,pp2]+dmm_rt
          
          for erz in eraz_arr:
            erz.eraz[lazt_id]=lazt_id
          eraz_arr.append(pp2)
        edb=sentUnit(']',fast=1)
        edb.id=lazt_id
        self.real_return.append(edb)
            

    if retThis:
      self.real_return=emb_and_v(self.msg,p_wgt=self.wgt,p_sta=self.upper,p_end=self.lower)+self.real_return[1:]
      
   

    if prepand is not None:
      self.real_return=[prepand]+self.real_return
 
  def nfo(self,trans_wgt=True,trans_sta=False,trans_end=False,extra=False):
    ret_wgt=self.wgt
    ret_sta=self.upper
    ret_end=self.lower
    if trans_wgt and ret_wgt is None:
      ret_wgt=1.0
    if trans_sta and ret_sta is None:
      ret_sta=0
    if trans_end and ret_end is None:
      ret_end=1.0
    if extra:
      return self.msg, ret_wgt, ret_sta, ret_end,[self.wgt is None,self.upper is None,self.lower is None]
    else:
      return self.msg, ret_wgt, ret_sta, ret_end

  def get_sig(self):
    ret=0
    if self.eraz:
      ret = 0x1000
    if self.upper is not None:
      return ret + 0x100
    if self.lower is not None:
      return ret + 0x100
    return ret

  def get_realstaend(self):
    sta=0
    endo=t_enc
    if self.upper is not None:
      sta=int(self.upper*t_enc +0.5)
    if self.lower is not None:
      endo=int(self.lower*t_enc +0.5)
    return sta, endo

  def get_realwgt(self):
    if self.wgt is None:
      return 1.0
    return self.wgt


  def set_emb(self,n):
    if n == 1:
      self.tok_len=1
      self.emb_wgt = self.emb_wgt1
      self.fast_emb= dedup[self.msg]
      self.fast_tkl=self.fast_emb.size(0)


  def emb_wgt1(self):
    wgg=torch.ones(self.fast_tkl)
    if self.wgt is not None:
      wgg*=self.wgt

    return self.fast_emb, wgg, torch.tensor([-1]*self.fast_emb.size(0))
  def emb_wgt0(self):
    tok=toker(self.msg)[1:-1]
    tkl=tok.size(0)
    self.tok_len=tkl
    wgg=torch.ones(tkl)
    if self.wgt is not None:
      wgg*=self.wgt
    amb = embedding(tok)
    return amb,wgg,tok

  def update_sta_end(self, wgt, sta, endo):
    if self.wgt is None:
      self.wgt=wgt
    if self.upper is None:
      self.upper=sta
    if self.lower is None:
      self.lower=endo

  def __repr__(self):
    ret=stringlizeNfo(self)
    if self.eraz:
      ret+='\n'+str(self.eraz)
    return ret


# arr=emb
class cond_getter:
  def __init__(self, arr, wgt_arr=None, reftxt=None, kndref=None, fast=-1, nsamp=1,cuda=True):
    self.txt=[]
    if reftxt is not None:
      self.txt=reftxt

    self.notSave=False
    self.add_sta=0
    self.d_sta=0
    self.is_simp=True
    self.get=self.get_simp
    self.get_txt=self.txt_simp

    self.vlm_count=0
    self.vlm_cond=self.vlm_cond0
    self.vlm_dup=self.vlm_dup0
    self.vlm_chunk=self.vlm_chunk0
    self.vlm_mix=self.vlm_mix0
    self.reset=self.reset0
    self.set_stas=self.set_stas0


    if arr is None:
      emb = get_empty()
      self.arr = cond_stage_model.from_emb(emb,nsamp=nsamp,cuda=cuda)
      return
    if fast==0:
      self.arr = from_emb(arr,wgt_arr=wgt_arr,nsamp=nsamp,cuda=cuda)
      return
    elif fast == 1:
      self.arr=arr
      return
  
    self.knd=kndref
    self.is_simp=False
    arr.append(arr[-1])
    self.arr=arr
    self.get=self.get_arr
    self.get_txt=self.txt_arr
      
  def get_knd(self):
    if self.is_simp:
      return np.ones(t_enc,dtype=np.uint8)*0xff
    return self.knd

  def get_fullarr(self):
    if self.is_simp:
      return [self.arr]*t_enc
    return self.arr


  def get_simp(self,d):
    return self.arr

  def size(self):
    if self.is_simp:
      arr = self.arr
    else:
      arr = self.arr[0]
    return arr.size(0), arr.size(1)

  def set_vlm(self,vlm_getters,vlm_wgt):
    self.vlm = vlm_getters
    self.vlm_count=len(vlm_getters)
    self.vlm_wgt = vlm_wgt.cuda()
    self.reset=self.reset_vlm
    self.set_stas=self.set_stas_vlm
    self.vlm_dup=self.vlm_dup_v
    self.vlm_cond=self.vlm_cond_v
    self.vlm_chunk=self.vlm_chunk_v
    self.vlm_mix=self.vlm_mix_v
    self.vlm_div=torch.sum(vlm_wgt).abs()

  def vlm_mix0(self,cond_list,cfg_r):
    return cond_list[0]*cfg_r

  def vlm_mix_v(self,cond_list,cfg_r):
    wgt = self.vlm_wgt*cfg_r/self.vlm_div
    mix=cond_list[0]*wgt[0]
    l=len(cond_list)
    for i in range(1,l):
      mix+=cond_list[i]*wgt[i]
    return mix

  def vlm_chunk0(self,ot):
    uncond, cond=ot.chunk(2)
    return uncond,[cond]


  def vlm_chunk_v(self,ot):
    foo=list(ot.chunk(self.vlm_count+1))
    return foo[0],foo[1:]


  def vlm_cond0(self,d):
    return torch.cat([ self.uc.get(d) , self.get(d) ])

  def vlm_cond_v(self,d):
    vcot=self.vlm_count
    kd=[None]* (1+vcot)
    kd[0]=self.uc.get(d)
    for i in range(vcot):
      kd[i+1]=self.vlm[i].get(d)
    return torch.cat(kd)

  def vlm_dup0(self,x):
    return torch.cat([x] * 2)

  def vlm_dup_v(self,x):
    return torch.cat([x] * (1+self.vlm_count) )

  def pad_and_setUC(self,padto=-1,extn=None,uc=None):
    if uc is not None:
      uc.set_subcond()
      self.uc=uc
    if self.vlm_count > 0:
      for v in self.vlm:
        v.pad(padto=padto,extn=extn)
      return
    self.pad(padto=padto,extn=extn)

  def pad(self,padto=-1,extn=None):
    global Max_EMB_len
    if padto<0:
      padto=Max_EMB_len
    if extn is None and padto==cond_stage_model.max_length:
      return
    padto=nearpower2(padto)
    b,origsiz=self.size()
    extnl=0
    if extn is not None:
      extnl=extn.size(0)
      allsiz=origsiz+extnl
      if allsiz > padto:
        padto=nearpower2(allsiz)
      if allsiz > Max_EMB_len:
        Max_EMB_len=allsiz

    if self.is_simp:
      self.arr=catnoise(self.arr,padto)
      if extnl > 0:
        self.arr[:,origsiz:allsiz]=extn.expand(b,-1,-1)
      return
    kndrev=dict()
    knd=self.knd
    knd_l=len(knd)
    arr=self.arr
    for i in range(knd_l):
      kndrev[knd[i]]=arr[i]

    if extnl > 0:
      extn-extn.expand(b,-1,-1)
      for k in kndrev:
        kaat=catnoise(kndrev[k],padto)
        kaat[:,origsiz:allsiz]=extn
        kndrev[k]=kaat
    else:
      for k in kndrev:
        kndrev[k]=catnoise(kndrev[k],padto)

    for i in range(knd_l):
      arr[i]=kndrev[knd[i]]

  def set_subcond(self):
    self.set_stas=self.set_stas_nouc
    self.reset=self.reset_nouc

  def set_stas_nouc(self,dsta=-1,addsta=-1):
    if dsta >=0:
      self.d_sta=dsta
    if addsta >=0:
      self.add_sta=addsta

  def set_stas0(self,dsta=-1,addsta=-1):
    self.set_stas_nouc(dsta,addsta)
    self.uc.set_stas(dsta,addsta)

  def set_stas_vlm(self,dsta=-1,addsta=-1):
    self.uc.set_stas(dsta,addsta)
    for v in self.vlm:
      v.set_stas(dsta,addsta)


  def reset_nouc(self):
    self.add_sta=0
    self.d_sta=0


  def reset0(self):
    self.reset_nouc()
    self.uc.reset()

  def reset_vlm(self):
    self.uc.reset()
    for v in self.vlm:
      v.reset()
  
  def clone(self,vlm=True):
    dst=cond_getter(arr=[55],fast=1)
    dst.txt = self.txt
    dst.notSave = self.notSave
    dst.add_sta = self.add_sta
    dst.d_sta = self.d_sta
    dst.is_simp = self.is_simp
    dst.get = self.get
    dst.get_txt = self.get_txt
    dst.vlm_count = self.vlm_count
    dst.vlm_cond = self.vlm_cond
    dst.vlm_dup = self.vlm_dup
    dst.vlm_chunk=self.vlm_chunk
    dst.vlm_mix=self.vlm_mix
    dst.reset = self.reset
    dst.set_stas = self.set_stas
    if vlm:
      dst.vlm=self.vlm
    return dst


  def txt_simp(self,d):
    return self.txt
  
  def get_arr(self,d):
    sd=d+self.add_sta
    if self.d_sta > 1:
      sd=int(0.5+d*self.d_sta)
    return self.arr[sd]

  def txt_arr(self,d):
    return self.txt[d]

  def save(self,pname='prmt'):
    if self.notSave:
      return None
    sv=dict()
    if self.is_simp:
      sv[0]=True
      savarr=self.arr[0]
      dfarr=None
    else:
      sv[0]=False
      sv[2]=self.knd
      savarr, dfarr = kndmax_diff(self.knd)
      knd_l=len(savarr)
      for i in range(knd_l):
        savarr[i]=self.arr[ savarr[i] ][0]

    sv[1]=savarr
    if self.txt is not None:
      pname+=self.get_txt(0)[:20]
    pname+='.compiled_prompt'
    torch.save(sv,pname)
    return dfarr



  def load(self,nsamp=1,cuda=True):
    sv=torch.load(self.arr[0],map_location=cudev)
    simp=sv[0]
    self.notSave=True
    
    self.get_txt=self.txt_simp
    self.txt='===secret==='
    if simp:
      self.get=self.get_simp
      self.arr=sv[1].expand(nsamp,-1,-1)
    else:
      self.knd=sv[2]
      self.get=self.get_arr
      karr=sv[1]
      knd_l=len(karr)
      for i in range(knd_l):
        z=karr[i]
        if cuda:
          z=z.cuda()
        else:
          z=z.cpu()
        karr[i]=z.expand(nsamp,-1,-1)

      knduse = resizeknd(self.knd)
      arr=[None]*t_enc
      for i in range(t_enc):
        arr[i]=karr[knduse[i]]
      arr.append(arr[-1])
      self.knd=knduse
      self.arr=arr
    self.is_simp=simp


def get_real_return(unit):
  grr=unit.real_return
  del unit.real_return
  return grr

def rdmIDfunc(yd):
  #print(yd[:2])
  return random.randint(0, 2**32)


def resizeknd(knd):
  ldl=len(knd)
  jd_sta=0
  if ldl > t_enc:
    jd_sta=ldl/t_enc
    knd2=knd
  elif ldl < t_enc:
    rpt=int(0.9999+t_enc/ldl)
    knd2=knd.repeat( rpt )
    jd_sta=ldl*rpt/t_enc
  knduse=knd
  if jd_sta!=0:
    knduse=[None]*(t_enc+1)
    for d in range(t_enc):
      sd=int(0.5+d*jd_sta)
      knduse[d]=knd2[sd]
    knduse=knduse[:-1]
  return knduse


def kndmax_diff(knd):
  curknd=knd[0]
  dfarr=[]
  knd_l=len(knd)
  kndict=dict()
  for i in range(knd_l):
    wua=knd[i]
    if wua != curknd:
      curknd=knd[i]
      dfarr.append(i)
    kndict[wua]=i
  revknd=[]
  for i in range(knd_l):
    if i in kndict:
      revknd.append(kndict[i])
    else:
      break
  return revknd, dfarr


def stringlizeNfo(src):
  msg,wgt,sta,endo = src.nfo()
  if wgt != 1.0:
    msg+='+'+str(wgt)
  sig=0
  if sta is not None:
    sig+=1
  if endo is not None:
    sig+=2

  if sig == 0:
    return msg
  elif sig==1:
    return msg+':'+str(int(0.5+sta*100))+':'
  elif sig==2:
    return msg+'::'+str(int(0.5+endo*100))
  elif sig==3:
    return msg+':'+str(int(0.5+sta*100))+':'+str(int(0.5+endo*100))




def mkInsertor_pstz(string):
  fna = 'UserEmb/'+string[1:]+'.txt'
  with open(fna,'rt') as f:
    stz=f.read().splitlines()
  stz=('@'.join(stz)).replace('@@','^').split('^')
  stz_l=len(stz)
  stz_n=[]
  for i in range(stz_l):
    txt=stz[i]
    if txt.startswith('##'):
      break
    if txt[0] == '#':
      continue
    if '@' in txt:
      stz2=txt.split('@')
      arr=[]
      for s in stz2:
        arr+=get_real_return(sentUnit(s))
      stz_n.append(arr)
    else:
      stz_n.append( get_real_return(sentUnit(stz[i])) )
  return stz_n




def i2t(strr, ifempty=None):
  if strr:
    f = float(strr)
    if f > 1:
      f/=100 
    return f
  return ifempty

def m2mw(strr,prev,p_wgt):
  wgt=p_wgt
  spl=strr.split('+')
  if len(spl)>1:
    wgt=float(spl[1])
    strr=spl[0]
    if strr == '':
      strr=prev
  return strr,wgt


InfoChrs='1234567890+-:. '
def findposiblesplit(str_in):
  lstr=len(str_in)-1
  for n in range(lstr,-1,-1):
    if str_in[n] not in InfoChrs:
      return n-lstr
  return 0

def mktaps(str,sep=';',p_wgt=None,p_sta=None,p_end=None):
  Enbale_s_in_s = True
  if sep != ';':
    Enbale_s_in_s=False
  segs=str.split(sep)
  if len(segs[-1]) == 0:
    segs=segs[:-1]
  if len(segs[0]) == 0:
    segs=segs[1:]
  prevstr=''
  ret=[]
  for s in segs:
    sta=p_sta
    endo=p_end
    repl_msg=None
    info_s=s
    s_in_s=False
    if Enbale_s_in_s and '|' in s:
      s_in_s=True
      idx=findposiblesplit(s)
      if idx == 0:
        info_s = 'dummy'
        repl_msg=s
      else:
        info_s = 'dummy'+s[idx:]
        repl_msg = s[:idx]

    msg=info_s.split(':')
    if len(msg) > 2:
      sta=i2t(msg[1],p_sta)
      endo=i2t(msg[2],p_end)
    msg, wgt=m2mw(msg[0],prevstr,p_wgt)
    if s_in_s:
      msg=repl_msg+msg[5:]

    msg=msg.strip()
    prevstr=msg
    ret.append((msg,wgt,sta,endo))
  return ret

def m2unit(data,dtal,mtx):
  ret=[]
  for i in range(dtal):
    if mtx[i] !=0xff:
      ret.append(data[i])
  return ret





def chkrealexist(key,src_n):
  if not src_n.repls:
    return False
  if key in src_n.repls:
    return True

  return False

def vintzproc(src,k,v):
  dtal=len(src)
  for n in range(dtal):
    if chkrealexist(k,src[n]):
      vinfo=src[n].repls[k]
      brd=vinfo.repl(src[n],v)
      if len(brd) == 1:
        src[n]=brd[0]
      else:
        src=src[:n]+brd+src[n+1:]
  return src

def recurflatten(seed,key_list):
  k=key_list[-1]
  n_pl=ActivedPromptVars[k].ll
  n_seed=len(seed)
  newseed=[]
  for i in range(n_seed):
    for v in range(n_pl):
      src=copy.deepcopy(seed[i])
      newseed.append( vintzproc(src,k,v) )
  if len(key_list) > 1:
    return recurflatten(newseed,key_list[:-1])
  else:
    return newseed

def ActivedPromptVarsByCplx():
  key_list=list(ActivedPromptVars.keys())
  kl=len(key_list)
  for n in range(kl):
    key=key_list[n]
    key_list[n]=('%08X'%ActivedPromptVars[key].cplxLevel(-1))+key
  key_list.sort()
  for n in range(kl):
    key_list[n]=key_list[n][8:]
  return key_list


def proc3d(data):
  key_list=ActivedPromptVarsByCplx()
  arr= recurflatten([data],key_list)
  arrl=len(arr)
  for i in range(arrl):
    arr[i]=trimgroup(arr[i])
  return arr


def proc1d(data):
  return [trimgroup(data)]


def trymakeemb(tag):
  if tag in dedup:
    return True
  if os.path.isfile('UserEmb/'+tag[1:-1]+'.bin'):
    insert(tag)
    return True
  return False
    




def dfind_emb(txt,poz,l,p_wgt,p_sta,p_end):
  i=poz
  while i < l:
    c=txt[i]
    i+=1
    if c == '>':
      sig=txt[poz-1:i]
      unit = sentUnit(sig,fast=0,p_wgt=p_wgt,p_sta=p_sta,p_end=p_end)
      valid=trymakeemb(sig)
      if valid:
        unit.set_emb(1)
      else:
        unit.wgt=-333
        unit.msg=sig[1:-1]
      return unit, 0 ,i
    

def dfind_v(txt,poz,l,p_wgt,p_sta,p_end):
  i=poz
  while i < l:
    c=txt[i]
    i+=1
    if c == '}':
      sig=txt[poz-1:i]
      unit = sentUnit('}',fast=0,p_wgt=p_wgt,p_sta=p_sta,p_end=p_end)
      dmm=vinfo(sig)
      if dmm.valid:
        unit.repls[sig]=dmm
      else:
        unit.wgt=-333
        unit.msg=sig[1:-1]
      return unit, 0 ,i

def dfind_v_dummy(txt,poz,l,p_wgt,p_sta,p_end):
  i=poz
  while i < l:
    c=txt[i]
    i+=1
    if c == '}':
      sig=txt[poz:i-1]
      unit = sentUnit(sig,fast=0,p_wgt=p_wgt,p_sta=p_sta,p_end=p_end)
      unit.wgt=-333
      return unit, 0 ,i

def dfind_head(txt,poz,l,p_wgt,p_sta,p_end):
  i=poz
  while i < l:
    c=txt[i]
    i+=1
    if c == '<':
      if i - poz > 1:
        unit= sentUnit(txt[poz:i-1].strip(),fast=0,p_wgt=p_wgt,p_sta=p_sta,p_end=p_end)
      else:
        unit= sentUnit('empty',fast=0,p_wgt=-666)
      return unit, 1 ,i
    elif c == '{':
      if i - poz > 1:
        unit= sentUnit(txt[poz:i-1].strip(),fast=0,p_wgt=p_wgt,p_sta=p_sta,p_end=p_end)
      else:
        unit= sentUnit('empty',fast=0,p_wgt=-666)
      return unit, 2 ,i
  
  fina=sentUnit(txt[poz:].strip(),fast=0,p_wgt=p_wgt,p_sta=p_sta,p_end=p_end)
  fina.wgt=-333
  return fina,0,l


def canmerge(ret):
  if len(ret) == 0:
    return False
  if len(ret[-1].msg) < 2:
    return False
  if ret[-1].msg[0] == '<':
    return False
  return True

def emb_and_v(txt,p_wgt=None,p_sta=None,p_end=None,enable3d=True):
  l=len(txt)
  i=0
  functbl=[dfind_head, dfind_emb, dfind_v]
  if not enable3d:
    functbl[2]=dfind_v_dummy

  finderfunc=dfind_head
  ret=[]
  while i < l:
    result, nfunc, i = finderfunc(txt,i,l,p_wgt,p_sta,p_end)
    finderfunc=functbl[nfunc]
    if result.wgt == -333:
      if canmerge(ret):
        ret[-1].msg+=' '+result.msg
      else:
        result.wgt=p_wgt
        ret.append(result)
    elif result.wgt != -666:
      ret.append(result)

  return ret





def dumbunit(txt,wgt):
  if wgt == 1:
    wgt = None
  return emb_and_v(txt,p_wgt=wgt)



def filltimeinfo(arr,sta,endo,wgtfix):
  if not arr:
    return 0
  for itm in arr:
    itm.wgt+=wgtfix 
    itm.upper=sta
    itm.lower=endo
  return 1

def pp_edb(sta,endo):
  prepand=sentUnit('[',fast=1, p_sta=sta,p_end=endo )
  lazt_id=prepand.id
  edb=sentUnit(']',fast=1)
  edb.id=lazt_id
  return [prepand],[edb]

def flattenretk(retk):
  ret=retk[0]
  retkl=len(retk)
  if retkl == 2:
    ret[0].wgt=float(retk[1][0].msg)
  elif retkl > 2:
    timeinfo=float(retk[2][0].msg)
    hazcot=0
    hazcot+=filltimeinfo(ret,None,timeinfo,0.1)
    hazcot+=filltimeinfo(retk[1],timeinfo,None,0.1)
    if hazcot > 1:
      pp, edb = pp_edb(None,timeinfo)
      ret=pp+ret+edb
      pp, edb = pp_edb(timeinfo,None)
      erz_id=pp[0].id
      ret[0].eraz[erz_id]=erz_id
      retk[1]=pp+retk[1]+edb

    ret+=retk[1]

  return ret



def parsedumbformat(txt,sta=0,l=-1,wgt=1,sqq=False):
  cut0=sta
  if l < 0:
    l=len(txt)
  retk=[[]]
  ptidx=0


  i=sta
  while i < l:
    c=txt[i]
    i+=1
    if c == '(':
      if i-cut0>1:
        retk[ptidx]+=dumbunit(txt[cut0:i-1],wgt)
      cut0, ret = parsedumbformat(txt,i,l,wgt+0.1,sqq=True)
      i=cut0
      retk[ptidx]+=ret
    elif c == '[':
      if i-cut0>1:
        retk[ptidx]+= dumbunit(txt[cut0:i-1],wgt) 
      cut0, ret = parsedumbformat(txt,i,l,wgt-0.1,sqq=True)
      i=cut0
      retk[ptidx]+=ret
    elif c == ')':
      if i-cut0>1:
        retk[ptidx]+= dumbunit(txt[cut0:i-1],wgt) 
      return i,flattenretk(retk)
    elif c == ']':
      if i-cut0>1:
        retk[ptidx]+= dumbunit(txt[cut0:i-1],wgt)
        return i,flattenretk(retk)
    elif sqq and c == ':':
      if i-cut0>1:
        retk[ptidx]+= dumbunit(txt[cut0:i-1],wgt) 
      cut0=i
      ptidx+=1
      retk.append([])


  retk=flattenretk(retk)
  if cut0<l:
    retk+= dumbunit(txt[cut0:],wgt) 
  return retk




def pmpmtx(data_in,nsamp=1,cuda=True,fromtxt=True,enable3d=True):
  if len(data_in[0]) == 0:
    return [cond_getter(None,nsamp=nsamp,cuda=cuda)]
  arr = pmpmtx_preproc(data_in,fromtxt=fromtxt,enable3d=enable3d)

  arrl=len(arr)
  for c in range(arrl):
    arr_for_getter, fastmode,txt, kndref = to_arr_for_getter(arr[c],nsamp=nsamp,cuda=cuda)
    arr[c]=cond_getter(arr_for_getter,fast=fastmode,reftxt=txt,kndref=kndref)
  return arr


def pmpmtx_preproc(data_in,fromtxt=True,enable3d=True):
  global ActivedPromptVars
  ActivedPromptVars=dict()
  arr=[]

  if fromtxt:
    if len(data_in) == 1:
      if '((' in data_in[0]:
        arr=parsedumbformat(data_in[0])
      else:
        arr= emb_and_v(data_in[0], enable3d=enable3d)
    else:
      for d in data_in:
        if d.startswith('##'):
          break
        if len(d)>0 and d[0] != '#':
          arr+=get_real_return(sentUnit(d))
  else:
    arr=data_in

  if enable3d and ActivedPromptVars:
    arr=proc3d(arr)
  else:
    ActivedPromptVars=dict()
    arr=proc1d(arr)
  return arr

  


def to_arr_for_getter(data,nsamp=1,cuda=True):
  dtal=len(data)
  cpy_ones=np.ones(t_enc,dtype=np.uint8)
  cpy_eraz=cpy_ones*0xff
  mtx=np.ones((dtal,t_enc),dtype=np.uint8)

  txtid=-1
  txtkole=[]
  notTime=True
  for i in range(dtal):
    dta_i=data[i]
    sig = dta_i.get_sig()
    if sig > 0xFF:
      notTime=False
      mtx[i]*=0xFF

      
      sta0, end0 = dta_i.get_realstaend()
      
      mtx[i][sta0:end0]=cpy_ones[sta0:end0]

      if sig > 0xfff:
        erazd=dta_i.eraz
        for k in erazd:
          sta1, end1=erazd[k].get_realstaend()
          mtx[i][sta1:end1]=cpy_eraz[sta1:end1]

  if notTime:
    emb, wgt, txt = mk_emb_wgt(data,dtal)
    arr = from_emb(emb,wgt_arr=wgt,nsamp=nsamp,cuda=cuda)
    return  arr, 1, txt, None #arr, fastmode, txt


  mtx=mtx.transpose((1,0))
  knd=np.ones(t_enc,dtype=np.uint8)
  ar2i=dict()
  i2txt=[]
  txtid=0
  for i in range(t_enc):
    sig=str(mtx[i].tobytes())[2:-1].replace('\\','')
    if sig in ar2i:
      i_sig=ar2i[sig]
    else:
      ar2i[sig]=txtid
      i2txt.append( m2unit(data,dtal,mtx[i]) )
      i_sig=txtid
      txtid+=1
    knd[i]=i_sig
  

  if knd.sum() == 0:
    emb, wgt, txt = mk_emb_wgt(i2txt[0])
    arr = from_emb(emb,wgt_arr=wgt,nsamp=nsamp,cuda=cuda)
    return  arr, 1, txt, None
  
  knd_arr=[None]*t_enc
  knd_arr_txt=[None]*t_enc
  enc_l=len(i2txt)

  txtk=[None]*enc_l
  for i in range(enc_l):
    emb, wgt, txt = mk_emb_wgt(i2txt[i])
    i2txt[i] = from_emb(emb,wgt_arr=wgt,nsamp=nsamp,cuda=cuda)
    txtk[i]=txt
  
  for i in range(t_enc):
    poo=knd[i]
    knd_arr[i]=i2txt[poo]
    knd_arr_txt[i]=txtk[poo]

  return  knd_arr, -1, knd_arr_txt, knd




def wgtfix0(wgt):
  if wgt is None:
    return None
  elif wgt > 2:
    return 1+0.1*wgt
  elif wgt < -2:
    return -1+0.1*wgt
  else:
    return wgt
  
  
def wgtfix(b):
  b.wgt=wgtfix0(b.wgt)
  return b


def trimdpth(dyp):
  ret=[]
  for i in range(9,-1,-1):
    if dyp[i]:
      ret+=list(dyp[i])
  return ret

def trimgroup(unit_arr):
  bdict=dict()
  stapoz=dict()
  

  ul=len(unit_arr)
  dyp=[]
  for i in range(10):
    dyp.append(set())
  depth=0
  clean_ret=[]
  for i in range(ul):
    b=unit_arr[i]
    bmsg=b.msg
    if len(bmsg) == 1:
      if bmsg == '[':
        depth+=1
        dyp[depth].add(b.id)
        bdict[b.id]=b
        stapoz[b.id]=[i+1,None]
      elif bmsg == ']':
        depth-=1
        stapoz[b.id][1]=i
    else:
      clean_ret.append(wgtfix(b))

  dyp=trimdpth(dyp)
  if len(dyp) == 0:
    return clean_ret
  
  for k in dyp:
    sta, endo =stapoz[k]
    bdict[k].wgt=endo-sta+1



  for k in dyp:
    sta, endo =stapoz[k]
    b=bdict[k]
    nfo=b.eraz
    isany=False

    for erzid in nfo:
      cur=bdict[erzid]
      b.eraz[erzid]=cur
      isany=True
      if cur.yetproc:
        sta2, endo2 =stapoz[erzid]
        _,_,cur_osta, cur_oendo = cur.nfo(trans_sta=True,trans_end=True)
        for i in range(sta2,endo2):
          msg,_, cmp_osta, cmp_oendo = unit_arr[i].nfo(trans_sta=True,trans_end=True)
          if msg != ']':
            if cmp_osta < cur_osta:
              cur_osta=cmp_osta
            if cmp_oendo > cur_oendo:
              cur_oendo = cmp_oendo
        cur.upper=cur_osta
        cur.lower=cur_oendo
        cur.yetproc=False
        

    if isany:
      mergedict(unit_arr,b.eraz,sta,endo)
      b.eraz=None


   
  return clean_ret




def tenzclamp(tenz,tolen=77):
  dup=int(0.9999+(tolen/tenz.size(0)))
  return torch.cat([tenz]*dup)[:tolen]

def prmt_vlm(stz,intp,cuda=True):
  global n_samples
  n_samples = 1
  l_stz=len(stz)>>1
  vlm=[None]*l_stz
  vlm_wgt_up=[None]*l_stz
  vlm_wgt_down=[None]*l_stz

  Noupdown=True
  for i in range(l_stz):
    subcond = makeCs(stz[2*i],1, cuda=cuda,enable3d=False )[0]
    subcond.set_subcond()
    vlm[i] = subcond
    wgt=stz[2*i + 1]
    if wgt[0] == ',':
      Noupdown=False
      wgt=wgt.replace(' ','').replace('\t','').split(',')
      vlm_wgt_up[i] = float(wgt[1])
      vlm_wgt_down[i] = float(wgt[2])
      if intp < 2:
        intp=int(vlm_wgt_down[i]-vlm_wgt_up[i])
    else:
      wgt = float(wgt)
      vlm_wgt_up[i] = wgt
      vlm_wgt_down[i] = wgt

  if Noupdown:
    intp=1

  vlm_wgt_up=torch.tensor(vlm_wgt_up, dtype=torch.float32)
  if intp < 2:
    dummy=cond_getter(arr=[55],fast=1)
    dummy.set_vlm(vlm,vlm_wgt_up)
    return [dummy]

  vlm_wgt_down=torch.tensor(vlm_wgt_down, dtype=torch.float32)
  rett=[None]*intp
  intp_s=intp-1
  for i in range(intp):
    dummy=cond_getter(arr=[55],fast=1)
    wgt = torch.lerp(vlm_wgt_up, vlm_wgt_down, i/intp_s)
    dummy.set_vlm(vlm,wgt)
    rett[i]=dummy
  return rett




def prmt_bin(binfna,nsamp=1,cuda=True):
  if '%' in binfna:
    bink=[]
    for i in range(78):
      nfna=binfna%i
      if os.path.isfile(nfna):
        bink.append( torch.tensor( np.fromfile(nfna,dtype=np.float32) ).reshape((-1,768)) )
    tenz = tenzclamp(torch.cat(bink))
  else:
    tenz = tenzclamp( torch.tensor( np.fromfile(binfna,dtype=np.float32) ).reshape((-1,768)) )

  tenz=tenz.expand(nsamp,-1,-1)
  if cuda:
    tenz=tenz.cuda()

  return [cond_getter(tenz,fast=1)]
  

def calcknd(knd_arr,ptxt):
  knd_arr = np.stack(knd_arr).transpose((1,0))
  ar2i=dict()
  i2txt=[]
  txtid=0
  hgt,prmpl=knd_arr.shape

  kndmap=np.ones(hgt,dtype=np.uint8)

  for i in range(hgt):
    sig=str(knd_arr[i].tobytes())[2:-1].replace('\\','')
    if sig in ar2i:
      i_sig=ar2i[sig]
    else:
      ar2i[sig]=txtid
      i2txt.append( i )
      i_sig=txtid
      txtid+=1
    kndmap[i]=i_sig


  stk=len(i2txt)
  for i in range(0,prmpl):
    stacking=[None]*stk
    ge=ptxt[i]

    for n in range(stk):
      stacking[n]=ge.get(i2txt[n])
    ptxt[i]=torch.stack(stacking)

  return kndmap

def kmapout(kndmap,calc_result):
  stk=kndmap.shape[0]
  cout2=[None]*stk

  for i in range(stk):
    cout2[i]=calc_result[ kndmap[i] ]
  return cout2

def prmt_avg(ptxt,pwgt,prmpl):
  knd_arr=[None]*prmpl
  cplx=False
  for i in range(prmpl):
    if not ptxt[i].is_simp:
      cplx=True
    knd_arr[i] = ptxt[i].get_knd()
  
  if cplx:
    kndmap =calcknd( knd_arr, ptxt )
    

    cout=ptxt[0]*pwgt[0]
    for i in range(1,prmpl):
      cout+=(ptxt[i]*pwgt[i])

    
    cout2=kmapout(kndmap,cout)
    

    return [ cond_getter( cout2,kndref=kndmap ) ]

  cout=ptxt[0].get(0)*pwgt[0]
  for i in range(1,prmpl):
    cout+=(ptxt[i].get(0)*pwgt[i])
  return [ cond_getter( cout,fast=1 )]


def prmt_dymc(stz,cuda):
  prmpl=len(stz)>>1
  ptxt=[]
  pstp=[0]
  stpsum=1
  for i in range(prmpl):
    ptxt.append(  makeCs(stz[2*i],1, cuda=cuda,enable3d=False )[0]  )
    soi=float(stz[2*i+1])
    stpsum+=soi
    pstp.append(  stpsum  )

  for i in range(prmpl):
    pstp[i+1]=int(0.5+(pstp[i+1]/stpsum)*t_enc)

  bs_knd=ptxt[0].get_knd().astype(np.uint16)
  bs_arr=ptxt[0].get_fullarr()
  for i in range(1,prmpl):
    cut0=pstp[i]
    bs_knd[cut0:]=ptxt[i].get_knd()[cut0:].astype(np.uint16)+0x100*i
    bs_arr[cut0:]=ptxt[i].get_fullarr()[cut0:]

  return [ cond_getter( bs_arr, kndref=bs_knd ) ]


def prmt_intp_cplx(ptxt,pstp,knd_arr,prmpl):
  kndmap =calcknd( knd_arr, ptxt )

  intpos=[]
  for vv in range(prmpl):
    c1=ptxt[vv]
    c2=ptxt[vv+1]
    stp=pstp[vv]
    for i in range(stp):
      cn= kmapout(kndmap, (c2*i+c1*(stp-i))/stp )
      intpos.append( cond_getter(cn, kndref=kndmap) )

  lztbk=pstp[-1]
  if lztbk > 1:
    c1=ptxt[prmpl]
    c2=ptxt[0]
    for i in range(lztbk):
      cn=kmapout(kndmap, (c2*i+c1*(lztbk-i))/lztbk )
      intpos.append( cond_getter(cn, kndref=kndmap) )
  else:
    cn = kmapout(kndmap,ptxt[-1])
    intpos.append( cond_getter(cn, kndref=kndmap) )
  return intpos

def prmt_intp(stz,cuda):
  prmpl=len(stz)>>1
  ptxt=[None]*prmpl
  pstp=[None]*prmpl
  knd_arr=[None]*prmpl
  cplx=False
  for i in range(prmpl):
    ge=makeCs(stz[2*i],1, cuda=cuda,enable3d=False )[0]
    knd_arr[i] = ge.get_knd()
    if not ge.is_simp:
      cplx=True
    ptxt[i]=  ge  
    pstp[i]=  int(stz[2*i+1])+1  
  prmpl-=1

  if cplx:
    return prmt_intp_cplx(ptxt,pstp,knd_arr,prmpl)
  
  intpos=[]
  for vv in range(prmpl):
    c1=ptxt[vv].get(0)
    c2=ptxt[vv+1].get(0)
    stp=pstp[vv]
    for i in range(stp):
      cn=(c2*i+c1*(stp-i))/stp
      intpos.append( cond_getter(cn,fast=1) )

  lztbk=pstp[-1]
  if lztbk > 1:
    c1=ptxt[prmpl].get(0)
    c2=ptxt[0].get(0)
    for i in range(lztbk):
      cn=(c2*i+c1*(lztbk-i))/lztbk
      intpos.append( cond_getter(cn,fast=1) )
  else:
    intpos.append(ptxt[-1])
  return intpos

def prmt_set(stz):
  global prompt
  global neg_prompt
  global n_samples
  global H
  global W
  global seed_size
  global seed
  global Sampler
  global Karras
  global KarrasRho
  global ddim_num_steps
  global t_enc
  global cfg_scale


  prompt=stz[0]
  neg_prompt=stz[1]
  if preimg is None and revpreimg is None:
    syz=stz[2].split('x')
    W=int(syz[-1])
    H=int(syz[-2])
    if len(syz) > 2:
      n_samples=int(syz[0])
  if stz[3][0]!='#':
    seed_size=stz[3].replace(' ','').split(',')
    sdl=len(seed_size)//3
    for i in range(sdl):
      seed_size[i*3]=int(seed_size[i*3])
      seed_size[i*3+1]=int(seed_size[i*3+1])
      seed_size[i*3+2]=float(seed_size[i*3+2])
    seed_size=seed_size[:sdl*3]
  seed=int(stz[4])
  if stz[5][0]!='#':
    syz=stz[5].replace(' ','').split(',')
    Sampler=syz[0]
    f_sampler()
    if len(syz) > 1:
      if syz[1]!='x':
        Karras=True
      else:
        Karras=False
      KarrasRho=float(syz[2])
  ddim_num_steps=int(stz[6])
  t_enc=ddim_num_steps
  if preimg is not None and strength < 1:
    t_enc = int(strength * ddim_num_steps)+1
  cfg_scale=float(stz[7])
  return makeCs(prompt)


def printprompts(detailed=False):
  k=0
  for c in c_list:
    tstr='PromptV'+str(k)+' at step'
    dfarr=None
    if SaveCompiledPrompt:
      dfarr=c.save(tstr)
    k+=1
    if c.txt:
      print(tstr+'0:')
      print(c.get_txt(0))
      if detailed and c.knd is not None:
        if dfarr is not None:
          for j in dfarr:
            print(tstr+str(j)+':')
            print(c.get_txt(j))
        else:
          knd=c.knd
          prev=knd[0]
          knd_l=len(knd)
          for j in range(knd_l):
            cur=knd[j]
            if cur != prev:
              prev=cur
              print(tstr+str(j)+':')
              print(c.get_txt(j))

depthLimit=10

def mergedict(unit_arr,b_eraz,sta,endo):
  for n in range(sta,endo):
    ue=unit_arr[n]
    if ue.id == 0:
      if ue.eraz:
        for k in b_eraz:
          ue.eraz[k]=b_eraz[k]
      else:
        ue.eraz=b_eraz

def txtErr(prmt0,msg):
  print(msg)
  prmt=prmt0.split('/')[-1][:-4]
  print('err prompt: '+prmt)
  return pmpmtx([prmt0],nsamp=n_samples,enable3d=False)


def cmdtype(cmd0):
  if cmd0.startswith('intp:'):
    return 1
  elif cmd0.startswith('dymc:'):
    return 2
  elif cmd0.startswith('set:'):
    return 3
  elif cmd0.startswith('vlm:'):
    return 4
  elif cmd0.startswith('mad:'):
    return 10
  elif cmd0.startswith('avg:'):
    return 11
  return 0

rtdir=''
def makeCs(prmt,depth=0,cuda=True,enable3d=True):
  global rtdir
  if prmt.endswith('.txt'):
    if depth > depthLimit:
      return txtErr(prmt,'Too many ref, probably circular reference.')
    if depth==0:
      rtdir=''
      try:
        rtdir=prmt[:prmt.rindex('/')+1]
      except:
        pass
    depth+=1
    if not os.path.isfile(prmt):
      prmt=rtdir+prmt
      if not os.path.isfile(prmt):
        return txtErr(prmt,'ref not found.')
    with open(prmt,'rt') as f:
      stz=f.read().splitlines()
    cmd=stz[0].replace(' ','').replace('\t','').split('/')
    cmd0=cmdtype(cmd[0])
    if cmd0 == 0:
      return pmpmtx(stz,nsamp=n_samples,cuda=cuda,enable3d=enable3d)
    elif cmd0 == 1:
      if depth > 1:
        return txtErr(stz[1],'do not intp in ref')
      return prmt_intp(stz[1:],cuda=cuda)
    elif cmd0 == 2:
      return prmt_dymc(stz[1:],cuda=cuda)
    elif cmd0 == 3:
      return prmt_set(stz[1:])
    elif cmd0 == 4:
      if depth > 1:
        return txtErr(stz[1],'do not vlm in ref')
      intpp=1
      uss=cmd[0].split(':')
      if len(uss) > 1:
        uss=int(uss[1])
        if uss > 1:
          intpp=uss
      return prmt_vlm(stz[1:],intpp,cuda=cuda)



    prmpl=(len(stz)-1)>>1
    stz=stz[1:]
    ptxt=[]
    pwgt=[]
    wgtsum=0
    for i in range(prmpl):
      ptxt.append(  makeCs(stz[2*i],depth, cuda=cuda,enable3d=False )[0]  )
      wgt=float(stz[2*i+1])
      wgtsum+=wgt
      pwgt.append(  wgt  )
    if cmd0 == 11:
      for i in range(prmpl):
        pwgt[i]=pwgt[i]/wgtsum
    
    return prmt_avg(ptxt,pwgt,prmpl)

  elif prmt.endswith('.bin'):
    return prmt_bin(prmt,nsamp=n_samples,cuda=cuda)
  elif prmt.endswith('.compiled_prompt'):
    kn=cond_getter([prmt],fast=1)
    kn.load(nsamp=n_samples,cuda=cuda)
    return [kn]
  else:
    return pmpmtx([prmt],nsamp=n_samples,cuda=cuda,enable3d=enable3d)
