#!/bin/bash

edge_path='./input/ali_data/user_edge.csv'
field_path='./input/ali_data/user_field.npy'
target_path='./input/ali_data/user_city.csv'

code_path='./GraphSAGE/main.py'

gpus=0

learning_rate=0.001
weight_decay=1e-4
dropout=0.3
batch=512

# sh sh/ali_city/GraphSAGE.sh

printf "\n#### learning_rate=$learning_rate, weight_decay=$weight_decay, dropout=$dropout ####\n"
CUDA_VISIBLE_DEVICES=$gpus python $code_path --epochs 9999 --batch $batch \
--lr $learning_rate --weight-decay $weight_decay --dropout $dropout \
--edge-path $edge_path  --field-path $field_path --target-path $target_path
