
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class BottleneckHybridGNN(torch.nn.Module):

    def __init__(self, num_features, num_classes):

        super(BottleneckHybridGNN, self).__init__()

        self.encoder = torch.nn.Linear(
            num_features,
            64
        )

        self.conv1 = GATConv(
            64,
            8,
            heads=8,
            dropout=0.6
        )

        self.conv2 = GATConv(
            64,
            num_classes,
            heads=1,
            concat=False,
            dropout=0.6
        )

        self.shortcut = torch.nn.Linear(
            num_features,
            num_classes
        )

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x_encoded = F.elu(
            self.encoder(x)
        )

        x1 = F.elu(
            self.conv1(x_encoded, edge_index)
        )

        x1 = F.dropout(
            x1,
            p=0.6,
            training=self.training
        )

        x2 = self.conv2(
            x1,
            edge_index
        )

        return F.log_softmax(
            x2 + self.shortcut(x),
            dim=1
        )
