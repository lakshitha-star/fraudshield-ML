import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FraudGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super(FraudGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

