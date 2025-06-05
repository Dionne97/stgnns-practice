import tsl
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns             # Heatmaps

from tsl.datasets import MetrLA
from tsl.data import SpatioTemporalDataset
from tsl.ops.connectivity import edge_index_to_adj
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, DiffConv
from einops.layers.torch import Rearrange
from tsl.data.preprocessing.scalers import StandardScaler
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE
from tsl.engines import Predictor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from tsl.nn.models.stgn import GraphWaveNetModel

# --------------------------------------------------
# Configuration / Hyper‑parameters
# --------------------------------------------------
HIDDEN_SIZE = 32
RNN_LAYERS = 1
GNN_KERNEL = 2

DATA_HORIZON = 12         # forecast steps → 12×5 min = 60 min
DATA_WINDOW = 12          # encoder window
DATA_STRIDE = 1
CONNECTIVITY_THRESHOLD = 0.1
VAL_RATIO = 0.1
TEST_RATIO = 0.2
BATCH_SIZE = 32


# --------------------------------------------------
# Utility helpers
# --------------------------------------------------

def print_matrix(matrix):
    if torch.is_tensor(matrix):
        matrix = matrix.cpu().detach().numpy()
    return pd.DataFrame(matrix)


def print_model_size(model):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    msg = f"Number of trainable parameters: {n:,}"
    print("=" * len(msg))
    print(msg)


# --------------------------------------------------
# DataModule
# --------------------------------------------------

def create_datamodule():
    """Load METR‑LA and wrap in TSL DataModule"""
    ds = MetrLA(root="./data")

    conn = ds.get_connectivity(
        threshold=CONNECTIVITY_THRESHOLD,
        include_self=False,
        normalize_axis=1,
        layout="edge_index",
    )

    torch_ds = SpatioTemporalDataset(
        target=ds.dataframe(),
        connectivity=conn,
        mask=ds.mask,
        horizon=DATA_HORIZON,
        window=DATA_WINDOW,
        stride=DATA_STRIDE,
    )

    scalers = {"target": StandardScaler(axis=(0, 1))}
    splitter = TemporalSplitter(val_len=VAL_RATIO, test_len=TEST_RATIO)

    dm = SpatioTemporalDataModule(
        dataset=torch_ds,
        scalers=scalers,
        splitter=splitter,
        batch_size=BATCH_SIZE,
    )
    dm.setup()
    print(f"Data prepared → nodes: {dm.n_nodes} | channels: {dm.n_channels}")
    return dm


# --------------------------------------------------
# Model definition
# --------------------------------------------------
class TimeThenSpaceModel(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 32, rnn_layers: int = 1, gnn_kernel: int = 2):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.node_embed = NodeEmbedding(n_nodes, hidden_size)
        self.time_nn = RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            n_layers=rnn_layers,
            cell="gru",
            batch_first=True,
            return_only_last_state=True,
        )
        self.space_nn = DiffConv(hidden_size, hidden_size, k=gnn_kernel)
        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange("b n (t f) -> b t n f", t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # x: [B, W, N, F]
        x = self.encoder(x)
        node_e = self.node_embed().unsqueeze(0).unsqueeze(0)  # [1,1,N,F]
        x = x + node_e

        h = self.time_nn(x)                     # [B*N, F]
        B, _, N, F = x.shape
        h = h.view(B, N, F)                     # [B, N, F]
        h = self.space_nn(h, edge_index, edge_weight)
        out = self.decoder(h)
        out = self.rearrange(out)
        return out


# --------------------------------------------------
# Plot helpers
# --------------------------------------------------

def plot_adjacency_heatmap(adj, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj, cmap="viridis")
    plt.title(f"({model_name.upper()}) Adjacency Matrix Heatmap")
    plt.tight_layout()
    fname = f"{model_name}_adjacency_heatmap.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved adjacency heatmap → {fname}")


def plot_metric_single(df: pd.DataFrame, metric: str, model_name: str):
    """Bar plot of a single metric vs horizon."""
    title = f"({model_name.upper()}) {metric.upper()} vs Prediction Horizon"
    plt.figure(figsize=(6, 5))
    df[metric].plot(kind="bar")
    plt.title(title)
    plt.xlabel("Horizon (min)")
    plt.ylabel(metric.upper())
    plt.xticks(rotation=0)
    plt.tight_layout()
    fname = f"{model_name}_{metric.lower()}_vs_horizon.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {title} → {fname}")


def plot_actual_vs_predicted(actual, pred, sensor, step, model_name, length=200):
    plt.figure(figsize=(14, 4))
    plt.plot(actual[:length, step, sensor, 0], label="Actual")
    plt.plot(pred[:length, step, sensor, 0], label="Predicted")
    plt.title(f"({model_name.upper()}) Sensor {sensor} · Horizon step {step}")
    plt.legend()
    plt.tight_layout()
    fname = f"sensor{sensor}_h{step}_{model_name}.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


# --------------------------------------------------
# Experiment runner
# --------------------------------------------------

def run_experiment(model, dm, model_name):
    """Generic function to run a training and testing experiment."""
    print("=" * 60)
    print(f"Running experiment for model: {model_name}")
    print("=" * 60)

    print_model_size(model)

    loss_fn = MaskedMAE()
    metrics = {
        "mae": MaskedMAE(),
        "mape": MaskedMAPE(),
        "mse": MaskedMSE(),
        "mae_at_15": MaskedMAE(at=2),
        "mae_at_30": MaskedMAE(at=5),
        "mae_at_60": MaskedMAE(at=11),
        "mse_at_15": MaskedMSE(at=2),
        "mse_at_30": MaskedMSE(at=5),
        "mse_at_60": MaskedMSE(at=11),
    }

    predictor = Predictor(
        model=model,
        optim_class=torch.optim.Adam,
        optim_kwargs={"lr": 1e-3},
        loss_fn=loss_fn,
        metrics=metrics,
    )

    logger = TensorBoardLogger("logs", name=model_name)
    ckpt_cb = ModelCheckpoint(
        dirpath=f"logs/{logger.name}/{logger.version}",
        save_top_k=1, monitor="val_mae", mode="min")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[ckpt_cb],
        limit_train_batches=10,
        limit_val_batches=5,
    )

    trainer.fit(predictor, datamodule=dm)

    predictor.freeze()
    test_res = trainer.test(predictor, datamodule=dm, ckpt_path="best", verbose=False)[0]
    print(f"Test metrics for {model_name}:", test_res)

    # --------------------------------------------------
    # Build performance DataFrame & plot each metric separately
    # --------------------------------------------------
    horizons = [15, 30, 60]
    perf = pd.DataFrame({
        "horizon": horizons,
        "mae": [test_res[f"test_mae_at_{h}"] for h in horizons],
        "mse": [test_res[f"test_mse_at_{h}"] for h in horizons],
        "mape": [test_res["test_mape"]]*3,
    }).set_index("horizon")

    print("\nAverage Performance Metrics Across All Sensors:")
    print(perf)

    for m in ["mae", "mse", "mape"]:
        plot_metric_single(perf, m, model_name)

    # --------------------------------------------------
    # Prediction sample plots
    # --------------------------------------------------
    preds = trainer.predict(predictor, dataloaders=dm.test_dataloader(), ckpt_path="best")
    preds = torch.cat([p if isinstance(p, torch.Tensor) else p["y_hat"] for p in preds], 0)
    actual = torch.cat([batch.y for batch in dm.test_dataloader()], 0)

    preds, actual = preds.cpu().numpy(), actual.cpu().numpy()
    steps = {"15min": 2, "30min": 5, "60min": 11}
    for s_id in range(min(3, dm.n_nodes)):
        for label, h in steps.items():
            if h < preds.shape[1]:
                plot_actual_vs_predicted(actual, preds, sensor=s_id, step=h, model_name=model_name)

    # --------------------------------------------------
    # Adjacency heatmap
    # --------------------------------------------------
    adj = edge_index_to_adj(dm.edge_index, dm.edge_weight).cpu().numpy()
    plot_adjacency_heatmap(adj, model_name)


# --------------------------------------------------
# Main
# --------------------------------------------------

def tts():
    """Run TimeThenSpaceModel experiment."""
    print(f"tsl {tsl.__version__} | torch {torch.__version__}")
    plt.style.use("seaborn-v0_8-whitegrid")

    dm = create_datamodule()

    model = TimeThenSpaceModel(input_size=dm.n_channels,
                               n_nodes=dm.n_nodes,
                               horizon=dm.horizon,
                               hidden_size=HIDDEN_SIZE,
                               rnn_layers=RNN_LAYERS,
                               gnn_kernel=GNN_KERNEL)
    
    run_experiment(model, dm, "tts")


def graphwavenet():
    """Run GraphWaveNetModel experiment."""
    dm = create_datamodule()
    model = GraphWaveNetModel(
        input_size=dm.n_channels,
        output_size=dm.n_channels,  # Assuming prediction of all input features
        horizon=DATA_HORIZON,
        n_nodes=dm.n_nodes,
        hidden_size=HIDDEN_SIZE,
        ff_size=256,                 # Default value
        n_layers=RNN_LAYERS,         # Using existing global RNN_LAYERS
        temporal_kernel_size=2,      # Default value
        spatial_kernel_size=GNN_KERNEL, # Using existing global GNN_KERNEL
        learned_adjacency=True,      # Default value
        emb_size=10,                 # Default value for learned_adjacency
        dilation=2,                  # Default value
        dilation_mod=2,              # Default value
        norm='batch',                # From use_batch_norm=True
        dropout=0.1,                 # Retained existing value
        exog_size=0                  # Default value
    )
    
    run_experiment(model, dm, "graphwavenet")


if __name__ == "__main__":
    tts()
    graphwavenet()
