#!/bin/bash

## Stage-level アーキテクチャを探索
# name=$1
# genotype=$2
# save=$3
# batch_size=64
# epoch=50
# train_portion=0.5 # searchの場合train_portionは0.5が最大値
# seed=0

# python searchStage_main.py \
    # --name $name\
    # --dataset CIFAR100\
    # --epochs $epoch\
    # --train_portion $train_portion\
    # --genotype $genotype\
    # --save $save \
    # --spec_cell

## Stage-level アーキテクチャを評価
# name=$1
# genotype=$2
# DAG=$3
# save=$4
# dataset=cifar100
# batch_size=64
# epoch=100
# train_portion=0.9
# seed=1
# python evaluateStage_main.py \
#     --name $name \
#     --genotype $genotype \
#     --DAG $DAG \
#     --dataset $dataset\
#     --batch_size $batch_size \
#     --epochs $epoch \
#     --train_portion $train_portion \
#     --seed $seed \
#     --save $save \
#     --spec_cell


## Stage-level アーキテクチャを評価
save=$1
genotype=$2
DAG=$3
path=$4
dataset=cifar100
batch_size=64
seed=0

python testStage_main.py \
    --save $save \
    --dataset $dataset \
    --batch_size $batch_size \
    --genotype $genotype \
    --DAG $DAG \
    --seed $seed \
    --resume_path $path \
    --spec_cell