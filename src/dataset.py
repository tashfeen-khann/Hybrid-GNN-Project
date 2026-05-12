
from torch_geometric.datasets import Planetoid, Amazon
import torch_geometric.transforms as T
import torch
import numpy as np


def create_masks(data):

    num_nodes = data.num_nodes

    indices = np.random.permutation(num_nodes)

    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)

    train_idx = indices[:train_size]
    val_idx = indices[
        train_size:train_size + val_size
    ]
    test_idx = indices[
        train_size + val_size:
    ]

    data.train_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool
    )

    data.val_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool
    )

    data.test_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool
    )

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data


def get_data(name):

    path = './data/' + name

    if name in [
        'Cora',
        'CiteSeer',
        'PubMed'
    ]:

        dataset = Planetoid(
            root=path,
            name=name,
            transform=T.NormalizeFeatures()
        )

        data = dataset[0]

    else:

        dataset = Amazon(
            root=path,
            name='Computers',
            transform=T.NormalizeFeatures()
        )

        data = dataset[0]

        data = create_masks(data)

    return dataset, data
