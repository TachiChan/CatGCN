import torch

import numpy as np
from parser import parameter_parser
from clustering import ClusteringMachine
from clustergnn import ClusterGNNTrainer

from utils import tab_printer, graph_reader, field_reader, target_reader

def main():
    """
    Parsing command line parameters, reading data, graph decomposition, fitting and scoring the model.
    """
    args = parameter_parser()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    tab_printer(args)
    graph = graph_reader(args.edge_path)
    field_index = field_reader(args.field_path)
    target = target_reader(args.target_path)
    clustering_machine = ClusteringMachine(args, graph, field_index, target)
    clustering_machine.decompose()
    gnn_trainer = ClusterGNNTrainer(args, clustering_machine)
    gnn_trainer.train_val_test()

if __name__ == "__main__":
    main()
