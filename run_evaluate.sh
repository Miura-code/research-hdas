#!/bin/bash


# ===== セルレベルアーキテクチャを評価　=====
# name=$1
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


## ===== 複数アーキテクチャを評価 =====
# name=$1
# save=$2
# dataset=cifar100
# batch_size=64
# epoch=100
# train_portion=0.9
# seed=0
# genotypes=(/home/miura/lab/research-hdas/results/search_cell/CIFAR100/BASELINE/EXP-20240713-020831/GENO/EP43-best.pickle /home/miura/lab/research-hdas/results/search_cell/CIFAR100/BASELINE/EXP-20240712-212432/GENO/EP49-best.pickle /home/miura/lab/research-hdas/results/search_cell/CIFAR100/BASELINE/EXP-20240712-163749/GENO/EP41-best.pickle /home/miura/lab/research-hdas/results/search_cell/CIFAR100/BASELINE/EXP-20240712-114919/GENO/EP45-best.pickle)
# for genotype in ${genotypes[@]}; do
#     echo ${genotype}
#     python evaluateCell_main.py \
#         --name $name \
#         --genotype $genotype \
#         --dataset $dataset\
#         --batch_size $batch_size \
#         --epochs $epoch \
#         --train_portion $train_portion \
#         --seed $seed \
#         --save $save
# done


# ===== セルレベルアーキテクチャをテスト　=====
save=test
genotype=$1
path=$2
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