#!/bin/bash


batch_size=128
epoch=100
train_portion=0.5
seed=0

python augmentStage_main.py \
    --name finetuning_test  \
    --batch_size $batch_size \
    --dataset cifar10 \
    --epochs $epoch \
    --genotype DARTS_V1 \
    --DAG HS_DAS_CIFAR \
    --train_portion $train_portion \
    --seed $seed \
