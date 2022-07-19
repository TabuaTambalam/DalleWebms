# DalleWebms

birds, mega model

![interpo bird1](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/bird1.gif?raw=true) ![interpo bird2](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/bird2.gif?raw=true)

maid knight, mega model

![interpo mk1](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/maidknightB.gif?raw=true)

![interpo mk2](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/maidknight0.gif?raw=true) ![interpo mk2plus](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/maidknight1.gif?raw=true)

rooms, mega model

![interpo mk2plus](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/room.gif?raw=true)

More: https://github.com/TabuaTambalam/DalleWebms/blob/main/rooms.md

interpo "superman", mini model

![interpo superman, mini model](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/interpo_mini.gif?raw=true)

# Outpainting:

![sabaku2](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/still/sbk0_2.png?raw=true)

![market](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/still/kof0_0.png?raw=true)
![market](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/still/kof0_1.png?raw=true)

![fighalf](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/still/genshn0.png?raw=true)

![bridge](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/still/brg4_0.png?raw=true)

![zelta](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/still/zlt0_2.png?raw=true)

- vq.param for NCNN here: https://github.com/TabuaTambalam/vqqncnn
can decode width-free image like those above. vq_vert.param can do height-free decode.

More samples here: https://github.com/TabuaTambalam/DalleWebms/tree/main/docs/still
- original seqs of all those samples: https://github.com/TabuaTambalam/DalleWebms/releases/download/0.1/seqs_outpainting.zip

# About Min_dall_singlfileGPU.ipynb

- Most codes came from https://github.com/kuprel/min-dalle , merged into single notebook, with following modifications:
- Use meta device for nnModule initializing, read more here: https://github.com/FrancescoSaverioZuppichini/Loading-huge-PyTorch-models-with-linear-memory-consumption
- Hardcoded to fp16 (the checkpoint itself is fp16, no need to upconv them to fp32 unless for speed reason)
- Partially-locked generation.
- Use ncnn VQGAN for decoding & interpolation, those animations above produced by ncnn VQGAN.
- Original seqs of those animation above is https://github.com/TabuaTambalam/DalleWebms/releases/download/0.1/ozv.bin
- colab link https://colab.research.google.com/github/TabuaTambalam/DalleWebms/blob/main/Min_dall_singlfileGPU.ipynb
