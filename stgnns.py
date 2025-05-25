import tsl
import torch
import numpy as np
import pandas as pd
from tsl.datasets import MetrLA
from tsl.ops.connectivity import edge_index_to_adj
from tsl.data import SpatioTemporalDataset

# =========================
# Utility Functions
# =========================
def print_matrix(matrix):
    return pd.DataFrame(matrix)

def print_model_size(model):
    tot = sum([p.numel() for p in model.parameters() if p.requires_grad])
    out = f"Number of model ({model.__class__.__name__}) parameters:{tot:10d}"
    print("=" * len(out))
    print(out)

# =========================
# Main Analysis
# =========================
def main():
    # Print library versions
    print(f"tsl version  : {tsl.__version__}")
    print(f"torch version: {torch.__version__}")

    # Set display options
    pd.options.display.float_format = '{:.2f}'.format
    np.set_printoptions(edgeitems=3, precision=3)
    torch.set_printoptions(edgeitems=2, precision=3)

    # Load dataset
    dataset = MetrLA(root='./data')
    print(dataset)

    # Dataset info
    print(f"Sampling period: {dataset.freq}")
    print(f"Has missing values: {dataset.has_mask}")
    print(f"Percentage of missing values: {(1 - dataset.mask.mean()) * 100:.2f}%")
    print(f"Has exogenous variables: {dataset.has_covariates}")
    print(f"Covariates: {', '.join(dataset.covariates.keys())}")
    print("Distance matrix:")
    print(print_matrix(dataset.dist))

    print("\nDataframe head:")
    print(dataset.dataframe().head())

    print(f"\nDefault similarity: {dataset.similarity_score}")
    print(f"Available similarity options: {dataset.similarity_options}")
    print("==========================================")

    # Similarity matrix
    sim = dataset.get_similarity("distance")  # or dataset.compute_similarity()
    print("Similarity matrix W:")
    print(print_matrix(sim))

    connectivity = dataset.get_connectivity(threshold=0.1,
                                        include_self=False,
                                        normalize_axis=1,
                                        layout="edge_index")
    
    edge_index, edge_weight = connectivity

    print(f'edge_index {edge_index.shape}:\n', edge_index)
    print(f'edge_weight {edge_weight.shape}:\n', edge_weight)

    print("==========================================")
    adj = edge_index_to_adj(edge_index, edge_weight)
    print(f'A {adj.shape}:')
    print_matrix(adj)

def build_pytorch_ready_dataset():
    # Get the edge index and edge weight
    dataset = MetrLA(root='./data')
    connectivity = dataset.get_connectivity(threshold=0.1,
                                        include_self=False,
                                        normalize_axis=1,
                                        layout="edge_index")
    torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                      connectivity=connectivity,
                                      mask=dataset.mask,
                                      horizon=12,
                                      window=12,
                                      stride=1)
    print(torch_dataset)
    sample = torch_dataset[0]
    print(sample)
    

if __name__ == "__main__":
   # main()
   build_pytorch_ready_dataset()