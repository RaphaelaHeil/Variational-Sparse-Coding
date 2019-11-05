#!/usr/bin/env zsh

conda activate vsc

ks=32
for ls in 8 32 64 96 128 160 200
do
  python train-convvsc.py --dataset fashion --latent-size $ls --do-not-resume --kernel-size $ks 2>&1 | tee convVsc_$ls.out
done


# run via:  zsh -i ./vscrunner.sh
