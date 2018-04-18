#!/bin/bash

for option in expdecay95 LRescape ROP7; do
    python train_thelast.py --lr_scheduler $option > ${option}_rnnsep0418.out
done


#   running multiple jobs on 1 GPU harms performance too bad even though it has enough VGA memory
#   this is caused by cpu-gpu scheduling overhead
#   thus running it in sequence is ideal
#   (i.e. training model 1, model 2, model 3 in a row)
