

UseV=  1
xtrkt = '35'

def wrk():
	dec(dumped_seqs[0],'img0.png')   #dumped_seqs[0]  #tseq

'''
#!cp /content/ozv.bin /content/pretrained/ozv.bin
!python ncnnde.py
'''


if UseV == 0:
	UseV=False
else:
	UseV=True


exna='.cpu.bin'
if UseV:
	exna='.vul.bin'



import numpy as np
from PIL import Image
import ncnn
net = ncnn.Net()
net.opt.use_vulkan_compute = UseV
net.opt.use_fp16_packed = False
net.opt.use_fp16_storage = False
net.opt.use_fp16_arithmetic = False
net.load_param("/tmp/vq.param")
net.load_model("/tmp/vq.bin")


def dec(seq,ona):
	with net.create_extractor() as ex:
		ex.input("in0", ncnn.Mat(seq).clone())
		hrr, out0 = ex.extract("out0")
	Image.fromarray(np.array(out0).astype(np.uint8)).save(ona)



def xdec(seq,dummy):
	with net.create_extractor() as ex:
		ex.input("in0", ncnn.Mat(seq).clone())
		hrr, out0 = ex.extract(xtrkt)
	np.array(out0).tofile('/content/z'+xtrkt+exna)


dumped_seqs=np.fromfile('/content/ozv.bin',dtype=np.uint16).astype(np.int32).reshape((-1,256))
tseq=np.hstack([dumped_seqs[0].reshape((16,16)),dumped_seqs[1].reshape((16,16))]).reshape(512)

wrk()


