
import os
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix

from src.model import BottleneckHybridGNN
from src.dataset import get_data


os.makedirs('results', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

datasets_list = [
    'Cora',
    'CiteSeer',
    'PubMed',
    'Computers'
]

all_results = {}

print("TRAINING STARTED")


for name in datasets_list:

    dataset, data = get_data(name)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    data = data.to(device)

    max_epochs = 400 if name in [
        'PubMed',
        'Computers'
    ] else 200

    run_accs = []

    print(f'\\nDATASET: {name}')

    best_model = None

    for run in range(1, 11):

        model = BottleneckHybridGNN(
            dataset.num_features,
            dataset.num_classes
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
            weight_decay=5e-4
        )

        best_val_acc = 0
        final_test_acc = 0

        for epoch in range(1, max_epochs + 1):

            model.train()

            optimizer.zero_grad()

            out = model(data)

            loss = F.nll_loss(
                out[data.train_mask],
                data.y[data.train_mask]
            )

            loss.backward()

            optimizer.step()

            model.eval()

            with torch.no_grad():

                logits = model(data)

                val_acc = (
                    logits[data.val_mask].argmax(1)
                    ==
                    data.y[data.val_mask]
                ).sum().item() / data.val_mask.sum().item()

                test_acc = (
                    logits[data.test_mask].argmax(1)
                    ==
                    data.y[data.test_mask]
                ).sum().item() / data.test_mask.sum().item()

                if val_acc > best_val_acc:

                    best_val_acc = val_acc

                    final_test_acc = test_acc

                    best_model = copy.deepcopy(
                        model.state_dict()
                    )

        run_accs.append(final_test_acc)

        print(
            f'Run {run:02d} | '
            f'Test Accuracy: {final_test_acc:.4f}'
        )

    mean_score = np.mean(run_accs)
    std_score = np.std(run_accs)

    all_results[name] = {
        'mean_accuracy': float(mean_score),
        'std_accuracy': float(std_score)
    }

    torch.save(
        best_model,
        f'checkpoints/{name}_model_weights.pth'
    )

    with open(
        f'results/{name}_metrics.json',
        'w'
    ) as f:

        json.dump(
            all_results[name],
            f,
            indent=4
        )

    print(
        f'{name} FINAL: '
        f'{mean_score:.4f} ± {std_score:.4f}'
    )


names = list(all_results.keys())

means = [
    all_results[n]['mean_accuracy']
    for n in names
]

stds = [
    all_results[n]['std_accuracy']
    for n in names
]

plt.figure(figsize=(10, 6))

bars = plt.bar(
    names,
    means,
    yerr=stds,
    capsize=10
)

plt.title('Final Comparison Study')

plt.ylabel('Accuracy')

plt.savefig(
    'results/final_comparison_study.png'
)

plt.close()

summary_df = pd.DataFrame(all_results).T

summary_df.to_csv(
    'results/training_log.csv',
    index=False
)

print("\\nALL OUTPUTS SAVED")
