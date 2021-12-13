#!/bin/bash

edge_path='./input/ali_data/user_edge.csv'
field_path='./input/ali_data/user_field.npy'
target_path='./input/ali_data/user_city.csv'

code_path='./main.py'

gpus=0

gnn_units='64,64,64'
gnn_hops=1
graph_layer='appnp' 

graph_refining='none'
grn_units='none'
bi_interaction='none'
nfm_units='none' 
aggr_style='none' 

# sh sh/ali_city/APPNP.sh

learning_rate=0.01
weight_decay=0.0
dropout=0.1

printf "\n#### learning_rate=$learning_rate, weight_decay=$weight_decay, dropout=$dropout ####\n"
CUDA_VISIBLE_DEVICES=$gpus python $code_path --seed 42 --epochs 9999 --weight-balanced True \
--learning-rate $learning_rate --weight-decay $weight_decay --dropout $dropout \
--graph-refining $graph_refining --aggr-pooling mean --grn-units $grn_units \
--bi-interaction $bi_interaction --nfm-units $nfm_units \
--graph-layer $graph_layer --gnn-hops $gnn_hops --gnn-units $gnn_units \
--aggr-style $aggr_style \
--edge-path $edge_path  --field-path $field_path --target-path $target_path