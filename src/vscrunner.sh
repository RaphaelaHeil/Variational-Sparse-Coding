#!/usr/bin/env zsh

conda activate vsc

for ls in 8 32 64 96 128 160 200
do
  python train-vsc.py --alpha 0.01 --epochs 20 --latent-size $ls --do-not-resume --dataset fashion 2>&1 | tee reports/vsc_$ls.out
  #echo $ls
done


# run via:  zsh -i ./vscrunner.sh
