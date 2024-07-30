#!/bin/bash

name=$1

# ===== セルレベルアーキテクチャを評価　=====
# genotype=$2
# save=$3
# dataset=cifar100
# batch_size=64
# epoch=100
# train_portion=0.9
# seed=0
# python evaluateCell_main.py \
#     --name $name \
#     --genotype $genotype \
#     --dataset $dataset\
#     --batch_size $batch_size \
#     --epochs $epoch \
#     --train_portion $train_portion \
#     --seed $seed \
#     --save $save

# ===== セルレベルアーキテクチャをテスト　=====
save=$1
genotype=$2
path=$3
dataset=cifar100
batch_size=64
seed=0

python testCell_main.py \
    --save $save \
    --dataset $dataset \
    --batch_size $batch_size \
    --genotype $genotype \
    --seed $seed \
    --resume_path $path