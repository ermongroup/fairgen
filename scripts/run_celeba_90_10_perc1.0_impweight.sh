#!/bin/bash

python3 train.py \
--shuffle --batch_size 128 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 4 \
--G_lr 2e-4 \
--D_lr 2e-4 \
--dataset CA64 \
--data_root ./data \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--save_every 1000 --test_every 1000 \
--num_best_copies 50 --num_save_copies 1 \
--loss_type hinge \
--seed 777 \
--num_epochs 150 --start_eval 40 \
--reweight 1 --alpha 1.0 \
--bias 90_10 \
--perc 1.0  \
--name_suffix celeba_90_10_perc1.0_impweight