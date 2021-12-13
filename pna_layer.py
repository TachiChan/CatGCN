from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing, GCNConv


class PNAConv(MessagePassing):
    """
    Pure neighborhood aggregation layer.
    """
    def __init__(self, K=1, cached=False, bias=True, **kwargs):
        super(PNAConv, self).__init__(aggr='add', **kwargs)
        
        self.K = K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight,
                                        dtype=x.dtype)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)

        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__,
                                         self.K)
