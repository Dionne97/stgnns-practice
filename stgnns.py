import tsl
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt  # Added for plotting
import seaborn as sns            # Added for heatmap

from tsl.datasets import MetrLA
from tsl.data import SpatioTemporalDataset  # Added as per tutorial
from tsl.ops.connectivity import edge_index_to_adj
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter  # Path from user's latest change
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, DiffConv
from einops.layers.torch import Rearrange
from tsl.data.preprocessing.scalers import StandardScaler  # Corrected path
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE
from tsl.engines import Predictor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# =========================
# Configuration / Hyperparameters
# =========================
HIDDEN_SIZE = 32
RNN_LAYERS = 1
GNN_KERNEL = 2  # For TimeThenSpaceModel

DATA_HORIZON = 12
DATA_WINDOW = 12
DATA_STRIDE = 1
CONNECTIVITY_THRESHOLD = 0.1
VAL_RATIO = 0.1
TEST_RATIO = 0.2
BATCH_SIZE = 32


# =========================
# Utility Functions
# =========================

def print_matrix(matrix):
    if torch.is_tensor(matrix):
        matrix = matrix.cpu().detach().numpy()
    return pd.DataFrame(matrix)


def print_model_size(model):
    tot = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = f"Number of model ({model.__class__.__name__}) parameters:{tot:10d}"
    print("=" * len(out))
    print(out)


# =========================
# Dataset Exploration (Optional)
# =========================

def explore_raw_dataset_properties():
    print("Exploring MetrLA Dataset Properties...")
    dataset = MetrLA(root="./data")
    print(dataset)
    print(f"Sampling period: {dataset.freq}")
    print(f"Has missing values: {dataset.has_mask}")
    print(f"Percentage of missing values: {(1 - dataset.mask.mean()) * 100:.2f}%")
    print(f"Has exogenous variables: {dataset.has_covariates}")
    if dataset.has_covariates:
        print(f"Covariates: {', '.join(dataset.covariates.keys())}")
    print("Distance matrix sample (first 5x5):")
    print(print_matrix(dataset.dist[:5, :5]))
    print("\nDataframe head:")
    print(dataset.dataframe().head())
    print(f"\nDefault similarity: {dataset.similarity_score}")
    print(f"Available similarity options: {dataset.similarity_options}")
    print("==========================================")
    sim = dataset.get_similarity("distance")
    print("Similarity matrix W (first 5x5):")
    print(print_matrix(sim[:5, :5]))

    connectivity_tuple = dataset.get_connectivity(
        threshold=CONNECTIVITY_THRESHOLD,
        include_self=False,
        normalize_axis=1,
        layout="edge_index",
    )
    edge_index, edge_weight = connectivity_tuple
    print(f"edge_index shape: {edge_index.shape}")
    print(f"edge_weight shape: {edge_weight.shape}")
    print("==========================================")
    adj = edge_index_to_adj(edge_index, edge_weight, n_nodes=dataset.n_nodes)
    print(f"Adjacency matrix A shape: {adj.shape}")
    print("Adjacency matrix A (first 5x5 from edge_index):")
    print_matrix(adj[:5, :5])
    print("Finished exploring MetrLA Dataset Properties.\n")


# =========================
# DataModule Creation
# =========================

def create_datamodule():
    """Loads the MetrLA dataset and prepares a SpatioTemporalDataModule."""
    print("Creating SpatioTemporalDataModule for MetrLA (Tutorial Style)...")

    raw_dataset = MetrLA(root="./data")

    connectivity = raw_dataset.get_connectivity(
        threshold=CONNECTIVITY_THRESHOLD,
        include_self=False,
        normalize_axis=1,
        layout="edge_index",
    )

    torch_dataset = SpatioTemporalDataset(
        target=raw_dataset.dataframe(),
        connectivity=connectivity,
        mask=raw_dataset.mask,
        horizon=DATA_HORIZON,
        window=DATA_WINDOW,
        stride=DATA_STRIDE,
    )
    print(f"SpatioTemporalDataset created: {torch_dataset}")

    scalers = {"target": StandardScaler(axis=(0, 1))}
    splitter = TemporalSplitter(val_len=VAL_RATIO, test_len=TEST_RATIO)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=BATCH_SIZE,
    )

    print("Setting up DataModule (fitting scalers, splitting data)...")
    dm.setup()

    print(
        f"DataModule setup complete. Nodes: {dm.n_nodes}, Input Channels: {dm.n_channels}"
    )
    print("Finished creating SpatioTemporalDataModule.\n")
    return dm


# =========================
# Model Definition
# =========================
class TimeThenSpaceModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_nodes: int,
        horizon: int,
        hidden_size: int = 32,
        rnn_layers: int = 1,
        gnn_kernel: int = 2,
    ):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)
        self.time_nn = RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            n_layers=rnn_layers,
            cell="gru",
            batch_first=True,
            return_only_last_state=True,
        )
        self.space_nn = DiffConv(
            in_channels=hidden_size, out_channels=hidden_size, k=gnn_kernel
        )
        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange("b n (t f) -> b t n f", t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # x: [batch, time_window, nodes, features]
        x_enc = self.encoder(x)
        node_emb = self.node_embeddings()  # [N, F_hidden]
        if x.ndim == 4:
            node_emb = node_emb.unsqueeze(0).unsqueeze(0)  # [1,1,N,F_hidden]
        x_emb = x_enc + node_emb

        # Feed through temporal RNN (TSL block accepts 4‑D)
        h_time = self.time_nn(x_emb)  # [B*N, F_hidden]
        b, _, n, f_hidden = x_emb.shape
        h_time = h_time.view(b, n, f_hidden)

        # Spatial GNN
        z = self.space_nn(h_time, edge_index, edge_weight)

        # Decode & reshape to horizon
        out = self.decoder(z)
        out = self.rearrange(out)
        return out


# =========================
# Plotting Functions
# =========================

def plot_adjacency_heatmap(adj_matrix, title="Adjacency Matrix Heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj_matrix, cmap="viridis")
    plt.title(title)
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.tight_layout()
    plt.savefig("adjacency_heatmap.png")
    plt.close()
    print(f"Saved {title} to adjacency_heatmap.png")


def plot_performance_metrics(metrics_df, title="Performance Metrics vs. Horizon"):
    metrics_df.plot(kind="bar", figsize=(12, 7))
    plt.title(title)
    plt.xlabel("Prediction Horizon (minutes)")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("performance_metrics.png")
    plt.close()
    print(f"Saved {title} to performance_metrics.png")


def plot_actual_vs_predicted(
    actual,
    predicted,
    sensor_idx,
    horizon_step=0,
    num_steps=200,
    title_prefix="Actual vs. Predicted",
):
    plt.figure(figsize=(15, 5))
    plt.plot(
        actual[:num_steps, horizon_step, sensor_idx, 0], label="Actual", marker="."
    )
    plt.plot(
        predicted[:num_steps, horizon_step, sensor_idx, 0], label="Predicted", marker="."
    )
    plt.title(
        f"{title_prefix} - Sensor {sensor_idx} - Horizon Step {horizon_step}"
    )
    plt.xlabel("Time Step (within test set)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    filename = f"actual_vs_predicted_sensor{sensor_idx}_h{horizon_step}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")


# =========================
# Main Execution
# =========================

def main():
    print(f"tsl version  : {tsl.__version__}")
    print(f"torch version: {torch.__version__}")
    print("------------------------------------------")
    pd.options.display.float_format = "{:.2f}".format
    np.set_printoptions(edgeitems=3, precision=3)
    torch.set_printoptions(edgeitems=2, precision=3)
    plt.style.use("seaborn-v0_8-whitegrid")

    # explore_raw_dataset_properties()  # Optional

    dm = create_datamodule()

    # Derive model parameters
    input_size = dm.n_channels
    n_nodes = dm.n_nodes
    model_horizon = dm.horizon

    print("Model Hyperparameters:")
    print(f"  Input Size: {input_size}")
    print(f"  Nodes     : {n_nodes}")
    print(f"  Horizon   : {model_horizon}")
    print("------------------------------------------")

    stgnn_model = TimeThenSpaceModel(
        input_size=input_size,
        n_nodes=n_nodes,
        horizon=model_horizon,
        hidden_size=HIDDEN_SIZE,
        rnn_layers=RNN_LAYERS,
        gnn_kernel=GNN_KERNEL,
    )
    print(stgnn_model)
    print_model_size(stgnn_model)

    # Loss & metrics
    loss_fn = MaskedMAE()
    metrics = {
        "mae": MaskedMAE(),
        "mape": MaskedMAPE(),
        "mae_at_15": MaskedMAE(at=2),
        "mae_at_30": MaskedMAE(at=5),
        "mae_at_60": MaskedMAE(at=11),
        "mse": MaskedMSE(),
        "mse_at_15": MaskedMSE(at=2),
        "mse_at_30": MaskedMSE(at=5),
        "mse_at_60": MaskedMSE(at=11),
    }

    predictor = Predictor(
        model=stgnn_model,
        optim_class=torch.optim.Adam,
        optim_kwargs={"lr": 0.001},
        loss_fn=loss_fn,
        metrics=metrics,
    )

    logger = TensorBoardLogger(
        save_dir="logs", name="stgnn_experiment", version="fixed_v1"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"logs/{logger.name}/{logger.version}/checkpoints",
        save_top_k=1,
        monitor="val_mae",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        limit_train_batches=10,
        limit_val_batches=5,
        callbacks=[checkpoint_callback],
    )

    print("Starting training...")
    trainer.fit(predictor, datamodule=dm)
    print("Training finished.")

    print("Testing best checkpoint...")
    predictor.freeze()
    test_results = trainer.test(
        predictor, datamodule=dm, ckpt_path="best", verbose=False
    )
    print("Test results:", test_results)

    # -------------------------------
    # PREDICTION & PLOTS SECTION FIXED
    # -------------------------------
    print("\nGenerating prediction-based visualisations...")
    raw_preds = trainer.predict(
        predictor, dataloaders=dm.test_dataloader(), ckpt_path="best"
    )

    pred_tensors = []
    for item in raw_preds:
        if isinstance(item, torch.Tensor):
            pred_tensors.append(item)
        elif isinstance(item, dict) and "y_hat" in item:
            pred_tensors.append(item["y_hat"])
    if not pred_tensors:
        print("No tensor predictions returned – skipping plots.")
        return

    predictions = torch.cat(pred_tensors, dim=0).cpu().numpy()  # [S,H,N,F]

    actual_tensors = []
    for batch in dm.test_dataloader():
        target = batch.y if torch.is_tensor(batch.y) else batch["y"]
        actual_tensors.append(target)
    actuals = torch.cat(actual_tensors, dim=0).cpu().numpy()

    n_samples = min(len(predictions), len(actuals))
    predictions, actuals = predictions[:n_samples], actuals[:n_samples]
    print(f"Predictions shape: {predictions.shape} | Actuals shape: {actuals.shape}")

    # Plotting
    horizon_steps = {"15min": 2, "30min": 5, "60min": 11}
    num_sensors_to_plot = 3
    plot_len = 200

    for sensor in range(min(num_sensors_to_plot, dm.n_nodes)):
        for label, h in horizon_steps.items():
            if h >= predictions.shape[1]:
                continue
            plot_actual_vs_predicted(
                actuals,
                predictions,
                sensor_idx=sensor,
                horizon_step=h,
                num_steps=plot_len,
                title_prefix=f"Sensor {sensor} · {label}",
            )

    # -------------------------------
    # Additional viz: adjacency & performance
    # -------------------------------
    if hasattr(dm, "edge_index") and hasattr(dm, "edge_weight"):
        adj = edge_index_to_adj(dm.edge_index, dm.edge_weight).cpu().numpy()
        plot_adjacency_heatmap(adj)

    horizons_minutes = [15, 30, 60]
    performance_data = {
        "Horizon (min)": horizons_minutes,
        "MAE": [
            test_results[0].get(f"test_mae_at_{h}", np.nan) for h in horizons_minutes
        ],
        "MAPE": [
            test_results[0].get("test_mape", np.nan) for _ in horizons_minutes
        ],
        "MSE": [
            test_results[0].get(f"test_mse_at_{h}", np.nan) for h in horizons_minutes
        ],
    }
    perf_df = pd.DataFrame(performance_data).set_index("Horizon (min)")
    print("\nAverage Performance Metrics:")
    print(perf_df)
    plot_performance_metrics(perf_df)


if __name__ == "__main__":
    main()
