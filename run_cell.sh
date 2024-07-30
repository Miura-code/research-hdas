#!/bin/bash

## ===== セルレベル探索　=====
name=$1
# batch_size=64
# epoch=50
# train_portion=0.5 # searchの場合train_portionは0.5が最大値
# seed=1
# python searchCell_main.py \
#     --name $name\
#     --dataset CIFAR100\
#     --epochs $epoch\
#     --train_portion $train_portion

## ===== 複数Seedでセルレベル探索　=====
name=$1
batch_size=64
epoch=50
train_portion=0.5 # searchの場合train_portionは0.5が最大値
seed=1
for seed in 1 2 3 4; do
    python searchCell_main.py \
            --name $name\
            --dataset CIFAR100\
            --epochs $epoch\
            --train_portion $train_portion\
            --seed $seed
done