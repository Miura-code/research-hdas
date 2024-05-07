#!/bin/bash

name=$1

stage_architecture=("HS_DAS_CIFAR" "HS_DAS_CIFAR_SKIP" "STAGE_HSDAS_V1" "STAGE_HSDAS_V2" "STAGE_HSDAS_V3" "STAGE_SHALLOW" "STAGE_MIDDLE" "STAGE_DEEP" "STAGE_DARTS" )

batch_size=128
epoch=100
train_portion=0.5
seed=0

## ステージの評価・ファインチューニング
# python augmentStage_main.py \
#     --name $name  \
#     --batch_size $batch_size \
#     --dataset cifar10 \
#     --epochs $epoch \
#     --genotype DARTS_V1 \
#     --DAG HS_DAS_CIFAR \
#     --train_portion $train_portion \
#     --seed $seed \

# # for arch in "${stage_architecture[@]}"; do
# #     echo $arch
# #     python augmentStage_main.py \
# #     --name $arch  \
# #     --batch_size $batch_size \
# #     --dataset cifar10 \
# #     --epochs $epoch \
# #     --genotype DARTS_V1 \
# #     --DAG $arch \
# #     --train_portion $train_portion \
# #     --seed $seed
# done


## ステージの探索
# python searchStage_main.py \
#     --name macro-cifar10-test \
#     --w_weight_decay 0.0027  \
#     --dataset cifar10 \
#     --batch_size 64 \
#     --workers 0  \
#     --genotype DARTS_V1

## ステージのテスト
arch="HS_DAS_CIFAR"
path=/home/miura/lab/research-hdas/results/augment_Stage/cifar/HS_DAS_CIFAR/checkpoint.pth.tar
python testStage_main.py \
    --name test \
    --dataset cifar10 \
    --batch_size 128 \
    --genotype DARTS_V1 \
    --DAG $arch \
    --seed $seed \
    --resume_path $path
