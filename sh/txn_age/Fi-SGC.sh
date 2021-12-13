#!/bin/bash

edge_path='./input/txn_data/user_edge.csv'
field_path='./input/txn_data/user_field.npy'
target_path='./input/txn_data/user_age.csv'

code_path='./main.py'

gpus=0

gnn_units='64'
gnn_hops=5
graph_layer='fignn'

graph_refining='none'
grn_units='none'
bi_interaction='none' 
nfm_units='none' 
aggr_style='none'

# sh sh/txn_age/Fi-SGC.sh

learning_rate=0.1
weight_decay=1e-5
dropout=0.0
num_steps=3

printf "\n#### learning_rate=$learning_rate, weight_decay=$weight_decay, dropout=$dropout ####\n"
CUDA_VISIBLE_DEVICES=$gpus python $code_path --seed 42 --epochs 9999 --weight-balanced True \
--learning-rate $learning_rate --weight-decay $weight_decay --dropout $dropout \
--graph-refining $graph_refining --aggr-pooling mean --grn-units $grn_units \
--bi-interaction $bi_interaction --nfm-units $nfm_units \
--graph-layer $graph_layer --gnn-hops $gnn_hops --gnn-units $gnn_units \
--aggr-style $aggr_style --num-steps $num_steps \
--edge-path $edge_path  --field-path $field_path --target-path $target_path 
