# Bottleneck Hybrid GNN for Node Classification

This project implements a high-performance **Graph Neural Network (GNN)** using PyTorch Geometric. It features a custom **Bottleneck Hybrid** architecture designed to classify nodes in complex networks like academic citations and e-commerce product graphs.

## 🚀 Key Features
* **Bottleneck Encoder:** A linear layer that compresses high-dimensional input features to reduce noise and prevent overfitting.
* **GAT Attention Mechanism:** Multi-head attention layers that allow the model to learn the specific importance of neighboring nodes.
* **Residual Shortcut:** A skip connection that carries raw features directly to the output layer, preventing information loss and stabilizing training.
* **Multi-Dataset Support:** Built-in compatibility for **Cora, CiteSeer, PubMed,** and **Amazon Computers**.

---

## 🏗️ Model Architecture

The `BottleneckHybridGNN` consists of the following stages:

1.  **Input Compression:** Features are passed through a `Linear` bottleneck with `ELU` activation.
2.  **Attention Layer 1:** A `GATConv` layer with 8 attention heads. 
3.  **Attention Layer 2:** A final `GATConv` layer that aggregates heads into class predictions.
4.  **Global Shortcut:** A parallel `Linear` connection adds raw input data to the final layer to preserve the "global" signal. 



## 🔍 Scientific Observation: Result Fluctuations
Please note that every time the training script is executed, there will be **minor fluctuations** in the final accuracy results (usually within a ±0.5% range). 

**Why does this happen?**
* **Random Initialization:** At the start of every run, the neural network's weights are initialized using random values (Glorot/Xavier initialization).
* **Dropout Variability:** The Dropout layers randomly de-activate different neurons in every epoch, leading to slight variations in the learning path.
* **Convergence:** Because of these different starting points, the optimizer (Adam) settles into slightly different "local minima," resulting in unique final accuracy scores for every simulation.

To ensure scientific integrity, this project uses a **10-Run Average** to report the final mean performance.
---

## 🛠️ Installation & Setup

You will need **Python 3.8+** and **PyTorch Geometric**. Run the following commands in your VS Code terminal:

```bash
# Install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html

# Install the main library and visualization tools
pip install torch-geometric matplotlib numpy