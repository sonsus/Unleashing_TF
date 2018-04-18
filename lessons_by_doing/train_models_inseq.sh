#!/bin/bash

for option in expdecay95 LRescape ROP7; do
    python train_thelast.py --lr_scheduler $option > ${option}_rnnsep0418.out
done
