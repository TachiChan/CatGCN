# CatGCN
This is our Pytorch implementation for the paper:

>Weijian Chen, Fuli Feng, Qifan Wang, Xiangnan He, Chonggang Song, Guohui Ling and Yongdong Zhang. [CatGCN: Graph Convolutional Networks with Categorical Node Features](https://arxiv.org/abs/2009.05303). 

## Citation 
If you want to use our codes and datasets in your research, please cite:
```
@article{CatGCN,
  author    = {Weijian Chen and
               Fuli Feng and
               Qifan Wang and
               Xiangnan He and
               Chonggang Song and
               Guohui Ling and
               Yongdong Zhang},
  title     = {CatGCN: Graph Convolutional Networks with Categorical Node Features},
  journal   = {CoRR},
  volume    = {abs/2009.05303},
  year      = {2020}
}
```
## Environment Requirement
The code has been tested running under Python 3.6.8. The required packages are as follows:
* pytorch == 1.1.0
* torch-geometric == 1.3.2
* torch-sparse == 0.4.3
* torch-cluster == 1.4.5
* torch-scatter == 1.4.0
* networkx == 2.3
* numpy == 1.16.3
* scikit-learn == 0.22.1
* texttable == 1.6.2

## Training and Evaluation
The description of commands has been clearly stated in the codes (see the 'parameter_parser' function in parser.py).
In addition, we provide scripts in the "sh" folder to reproduce the results in the paper, including the baseline methods.

The processed datasets can be downloaded [here](https://drive.google.com/file/d/1N5dLm0jseRdD7Tly6F4GYJFGvRmdHH_M/view?usp=sharing), and the corresponding [process files](https://github.com/TachiChan/Data_Process) are also provided.

Running commands of CatGCN are as follows:
* Tencent-age, CatGCN
```
CUDA_VISIBLE_DEVICES=0 python main.py \
--learning-rate 0.1 --weight-decay 1e-4 --dropout 0.3 --diag-probe 1 \
--graph-refining agc --aggr-pooling mean --grn-units none \
--bi-interaction nfm --nfm-units none \
--graph-layer pna --gnn-hops 6 --gnn-units none \
--aggr-style sum --balance-ratio 0.4  \
--edge-path './input/txn_data/user_edge.csv'  --field-path './input/txn_data/user_field.npy' --target-path './input/txn_data/user_age.csv'
```

* Alibaba-purchase, CatGCN
```
CUDA_VISIBLE_DEVICES=0 python main.py \
--learning-rate 0.1 --weight-decay 1e-5 --dropout 0.3 --diag-probe 39 \
--graph-refining agc --aggr-pooling mean --grn-units none \
--bi-interaction nfm --nfm-units 64,64,64,64 \
--graph-layer pna --gnn-hops 8 --gnn-units none \
--aggr-style sum --balance-ratio 0.9  \
--edge-path './input/ali_data/user_edge.csv'  --field-path './input/ali_data/user_field.npy' --target-path './input/ali_data/user_buy.csv'
```

* Alibaba-city, CatGCN
```
CUDA_VISIBLE_DEVICES=0 python main.py \
--learning-rate 0.1 --weight-decay 1e-5 --dropout 0.9 --diag-probe 41 \
--graph-refining agc --aggr-pooling mean --grn-units 64,64 \
--bi-interaction nfm --nfm-units none \
--graph-layer pna --gnn-hops 3 --gnn-units none \
--aggr-style sum --balance-ratio 0.3  \
--edge-path './input/ali_data/user_edge.csv'  --field-path './input/ali_data/user_field.npy' --target-path './input/ali_data/user_city.csv'
```

Note that the results maybe fluctuate due to [the inherent randomness](https://pytorch.org/docs/stable/notes/randomness.html).

## Acknowledgment
Thanks to the Cluster-GCN implementation:
* [Cluster-GCN](https://github.com/benedekrozemberczki/ClusterGCN)
