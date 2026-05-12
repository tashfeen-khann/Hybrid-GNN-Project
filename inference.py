
import torch

from src.model import BottleneckHybridGNN
from src.dataset import get_data


dataset_name = 'Computers'

dataset, data = get_data(dataset_name)

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

data = data.to(device)

model = BottleneckHybridGNN(
    dataset.num_features,
    dataset.num_classes
).to(device)

model.load_state_dict(
    torch.load(
        f'checkpoints/{dataset_name}_model_weights.pth'
    )
)

model.eval()

with torch.no_grad():

    out = model(data)

    pred = out.argmax(dim=1)

    test_acc = (
        pred[data.test_mask]
        ==
        data.y[data.test_mask]
    ).sum().item() / data.test_mask.sum().item()

print(f'Test Accuracy: {test_acc:.4f}')
