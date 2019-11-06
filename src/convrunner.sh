#!/usr/bin/env zsh

conda activate vsc

ks='32,32'
for ls in 8 32 64 96 128 160 200
do
  #echo reports/convVsc_${ls}_${ks}.out
  python train-convvsc.py --dataset fashion --latent-size $ls --do-not-resume --alpha 0.01 --epochs 20 --kernel-size $ks 2>&1 | tee reports/convVsc_${ls}.out
done


# run via:  zsh -i ./vscrunner.sh
