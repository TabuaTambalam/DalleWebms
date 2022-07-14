# DalleWebms

birds, mega model

![interpo bird1](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/bird1.gif?raw=true) ![interpo bird2](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/bird2.gif?raw=true)

maid knight, mega model

![interpo mk1](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/maidknightB.gif?raw=true)

![interpo mk2](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/maidknight0.gif?raw=true) ![interpo mk2plus](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/maidknight1.gif?raw=true)

interpo "superman", mini model

![interpo superman, mini model](https://github.com/TabuaTambalam/DalleWebms/blob/main/docs/interpo_mini.gif?raw=true)

# About Min_dall_singlfileGPU.ipynb

- Most codes came from https://github.com/kuprel/min-dalle , merged into single notebook, with following modifications:
- Use meta device for initializing nnModule, read more here: https://github.com/FrancescoSaverioZuppichini/Loading-huge-PyTorch-models-with-linear-memory-consumption
- Hardcoded to fp16 (the checkpoint itself is fp16, no need to upconv them to fp32 unless for speed reason)
- Partially-locked generation.
- Use ncnn VQGAN for decoding & interpolation, those animations above produced by ncnn VQGAN.
- Original seqs of those animation above is https://github.com/TabuaTambalam/DalleWebms/releases/download/0.1/ozv.bin
