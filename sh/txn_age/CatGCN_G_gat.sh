#!/bin/bash

edge_path='./input/txn_data/user_edge.csv'
field_path='./input/txn_data/user_field.npy'
target_path='./input/txn_data/user_age.csv'

code_path='./main.py'

gpus=0

gnn_units='none'
gnn_hops=4
graph_layer='pna'

graph_refining='gat'
gat_units='64,64'
multi_heads='1' 
grn_units='none'
bi_interaction='none'
nfm_units='none' 
aggr_style='none' 

# sh sh/txn_age/CatGCN_G_gat.sh

learning_rate=0.1
weight_decay=1e-4
dropout=0.3

printf "\n#### learning_rate=$learning_rate, weight_decay=$weight_decay, dropout=$dropout ####\n"
CUDA_VISIBLE_DEVICES=$gpus python $code_path --seed 42 --epochs 9999 --weight-balanced True \
--learning-rate $learning_rate --weight-decay $weight_decay --dropout $dropout \
--graph-refining $graph_refining --aggr-pooling mean --grn-units $grn_units \
--bi-interaction $bi_interaction --nfm-units $nfm_units \
--graph-layer $graph_layer --gnn-hops $gnn_hops --gnn-units $gnn_units \
--aggr-style $aggr_style --gat-units $gat_units --multi-heads $multi_heads \
--edge-path $edge_path  --field-path $field_path --target-path $target_path