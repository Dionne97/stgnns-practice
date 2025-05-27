import tsl
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt # Added for plotting
import seaborn as sns           # Added for heatmap

from tsl.datasets import MetrLA
from tsl.data import SpatioTemporalDataset
from tsl.ops.connectivity import edge_index_to_adj
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, DiffConv
from einops.layers.torch import Rearrange  # reshape data with Einstein notation# from tsl.data.datamodule import PandasDataModule
from tsl.data.scalers import StandardScaler
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE # Added MaskedMSE
from tsl.engines import Predictor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# =========================
# Configuration / Hyperparameters
# =========================
# Model Hyperparameters
HIDDEN_SIZE = 32
RNN_LAYERS = 1
GNN_KERNEL = 2

# Data Preprocessing Parameters
DATA_HORIZON = 12  # Prediction horizon 12 (60 minutes)
DATA_WINDOW = 12   # Window size = 12 (60 minutes)
DATA_STRIDE = 1
CONNECTIVITY_THRESHOLD = 0.1
VAL_RATIO = 0.1    # Holdout validation 0.1
TEST_RATIO = 0.2   # Holdout validation 0.2 (train will be 0.7)
BATCH_SIZE = 32    # Added batch size

# =========================
# Utility Functions
# =========================
def print_matrix(matrix):
    """Prints a matrix or tensor as a Pandas DataFrame."""
    if torch.is_tensor(matrix):
        matrix = matrix.cpu().detach().numpy() # Added detach()
    return pd.DataFrame(matrix)

def print_model_size(model):
    """Prints the number of trainable parameters in a PyTorch model."""
    tot = sum([p.numel() for p in model.parameters() if p.requires_grad])
    out = f"Number of model ({model.__class__.__name__}) parameters:{tot:10d}"
    print("=" * len(out))
    print(out)

# =========================
# Dataset Exploration (Optional)
# =========================
def explore_raw_dataset_properties():
    """Loads and prints properties of the raw MetrLA dataset."""
    print("Exploring MetrLA Dataset Properties...")
    dataset = MetrLA(root='./data')
    print(dataset)

    # Dataset info
    print(f"Sampling period: {dataset.freq}")
    print(f"Has missing values: {dataset.has_mask}")
    print(f"Percentage of missing values: {(1 - dataset.mask.mean()) * 100:.2f}%")
    print(f"Has exogenous variables: {dataset.has_covariates}")
    if dataset.has_covariates:
        print(f"Covariates: {', '.join(dataset.covariates.keys())}")
    print("Distance matrix sample (first 5x5):")
    print(print_matrix(dataset.dist[:5, :5])) # Print a smaller sample

    print("\nDataframe head:")
    print(dataset.dataframe().head())

    print(f"\nDefault similarity: {dataset.similarity_score}")
    print(f"Available similarity options: {dataset.similarity_options}")
    print("==========================================")

    # Similarity matrix
    sim = dataset.get_similarity("distance")
    print("Similarity matrix W (first 5x5):")
    print(print_matrix(sim[:5, :5])) # Print a smaller sample
    
    connectivity = dataset.get_connectivity(threshold=CONNECTIVITY_THRESHOLD,
                                            include_self=False,
                                            normalize_axis=1,
                                            layout="edge_index")
    
    edge_index, edge_weight = connectivity
    print(f'edge_index shape: {edge_index.shape}')
    # print("Sample of edge_index can be printed here.") # Sample of edge_index
    print(f'edge_weight shape: {edge_weight.shape}')
    # print("Sample of edge_weight can be printed here.")    # Sample of edge_weight

    print("==========================================")
    adj = edge_index_to_adj(edge_index, edge_weight, n_nodes=dataset.n_nodes)
    print(f'Adjacency matrix A shape: {adj.shape}')
    print("Adjacency matrix A (first 5x5 from edge_index):")
    print_matrix(adj[:5, :5]) # Print a smaller sample
    print("Finished exploring MetrLA Dataset Properties.\n") # Added newline for spacing


# =========================
# DataModule Creation
# =========================
def create_datamodule():
    """Loads the MetrLA dataset and prepares a PandasDataModule."""
    print("Creating PandasDataModule for MetrLA...")
    raw_dataset = MetrLA(root='./data')
    
    # IMPORTANT: Get connectivity to pass to PandasDataModule
    # This connectivity will be used internally by the SpatioTemporalDataset within the DataModule
    connectivity = raw_dataset.get_connectivity(threshold=CONNECTIVITY_THRESHOLD,
                                                include_self=False, # As per previous setup
                                                normalize_axis=1,   # As per previous setup
                                                layout="edge_index")

    dm = PandasDataModule(
        df=raw_dataset.dataframe(),
        mask=raw_dataset.mask,
        connectivity=connectivity, # Pass connectivity here
        target_cols=None, # None means all columns are targets
        
        # Splitting configuration
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        
        # Windowing and horizon
        window=DATA_WINDOW,
        horizon=DATA_HORIZON,
        stride=DATA_STRIDE,
        
        # Scaler configuration
        scalers={'target': StandardScaler()}, # Apply standard scaler to target
        
        # Dataloader configuration
        batch_size=BATCH_SIZE,
        num_workers=0 # Adjust as needed
    )
    
    print(f"DataModule created. Shapes: dm.train_data.shape={dm.train_data.shape if dm.train_data is not None else 'N/A'}, dm.n_nodes={dm.n_nodes}, dm.n_channels={dm.n_channels}")
    # The connectivity (edge_index, edge_weight) is now part of the samples generated by the datamodule's internal dataset.
    # So, we don't need to return it separately if the model's forward pass takes the whole batch.
    print("Finished creating PandasDataModule.\n")
    return dm

# =========================
# Model Definition
# =========================
class TimeThenSpaceModel(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 32,
                 rnn_layers: int = 1,
                 gnn_kernel: int = 2):
        super(TimeThenSpaceModel, self).__init__()

        self.encoder = nn.Linear(input_size, hidden_size)
        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)
        self.time_nn = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers,
                           cell='gru',
                           return_only_last_state=True)
        self.space_nn = DiffConv(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 k=gnn_kernel)
        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, batch): # Changed to accept batch object
        # x: [batch, time_window, nodes, features]
        # edge_index, edge_weight are expected to be in batch from DataModule
        x = batch.x
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        
        x_enc = self.encoder(x)
        node_emb = self.node_embeddings().unsqueeze(0).unsqueeze(0)
        x_emb = x_enc + node_emb
        
        b, t, n, f = x_emb.shape
        x_rnn_in = x_emb.permute(0, 2, 1, 3).reshape(b * n, t, f)
        h_rnn_out = self.time_nn(x_rnn_in)
        
        h = h_rnn_out.reshape(b, n, f)
        z = self.space_nn(h, edge_index, edge_weight)
        
        x_out = self.decoder(z)
        x_horizon = self.rearrange(x_out)
        return x_horizon

# =========================
# Plotting Functions
# =========================
def plot_adjacency_heatmap(adj_matrix, title="Adjacency Matrix Heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj_matrix, cmap="viridis")
    plt.title(title)
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.savefig("adjacency_heatmap.png")
    print(f"Saved {title} to adjacency_heatmap.png")
    plt.close()

def plot_performance_metrics(metrics_df, title="Performance Metrics vs. Horizon"):
    metrics_df.plot(kind='bar', figsize=(12, 7))
    plt.title(title)
    plt.xlabel("Prediction Horizon (minutes)")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("performance_metrics.png")
    print(f"Saved {title} to performance_metrics.png")
    plt.close()

def plot_actual_vs_predicted(actual, predicted, sensor_idx, horizon_step=0, num_steps=200, title_prefix="Actual vs. Predicted"):
    plt.figure(figsize=(15, 5))
    plt.plot(actual[:num_steps, horizon_step, sensor_idx, 0], label="Actual", marker='.')
    plt.plot(predicted[:num_steps, horizon_step, sensor_idx, 0], label="Predicted", marker='.')
    plt.title(f"{title_prefix} - Sensor {sensor_idx} - Horizon Step {horizon_step} (e.g., 15min ahead if step=2)")
    plt.xlabel("Time Step (within test set sample)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    filename = f"actual_vs_predicted_sensor{sensor_idx}_h{horizon_step}.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

# =========================
# Main Execution
# =========================
def main():
    """Main function to run the STGNN setup, training, testing, and visualization."""
    print(f"tsl version  : {tsl.__version__}")
    print(f"torch version: {torch.__version__}")
    print("------------------------------------------")

    pd.options.display.float_format = '{:.2f}'.format
    np.set_printoptions(edgeitems=3, precision=3)
    torch.set_printoptions(edgeitems=2, precision=3)
    plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style for plots

    # --- 1. Explore Raw Dataset (Optional) ---
    # explore_raw_dataset_properties()

    # --- 2. Create DataModule ---
    dm = create_datamodule()
    
    # --- 3. Define Model Parameters ---
    # Parameters are now fetched from the DataModule
    input_size = dm.n_channels 
    n_nodes = dm.n_nodes
    # horizon for model output is same as DATA_HORIZON used in DataModule
    # It's also available as dm.horizon if needed, but we used DATA_HORIZON for model init.

    print(f"Model Hyperparameters (from DataModule & Config):")
    print(f"  Input Size (channels): {input_size}")
    print(f"  Number of Nodes: {n_nodes}")
    print(f"  Output Horizon: {DATA_HORIZON}") # Used for model
    print(f"  Window Size: {DATA_WINDOW}")   # Used for data
    print(f"  Hidden Size: {HIDDEN_SIZE}")
    print(f"  RNN Layers: {RNN_LAYERS}")
    print(f"  GNN Kernel Size: {GNN_KERNEL}")
    print("------------------------------------------")

    # --- 4. Instantiate Model ---
    print("Instantiating TimeThenSpaceModel...")
    stgnn_model = TimeThenSpaceModel(input_size=input_size,
                                 n_nodes=n_nodes,
                                 horizon=DATA_HORIZON, # Model's output horizon
                                 hidden_size=HIDDEN_SIZE,
                                 rnn_layers=RNN_LAYERS,
                                 gnn_kernel=GNN_KERNEL)
    print(stgnn_model)
    print_model_size(stgnn_model)
    print("Model instantiated successfully.")

    # --- 5. Setup Predictor, Logger, Callbacks, and Trainer ---
    loss_fn = MaskedMAE()
    metrics = {
        'mae': MaskedMAE(),
        'mape': MaskedMAPE(),
        'mse': MaskedMSE(), # Added MSE for overall performance
        'mae_at_15min': MaskedMAE(at=2), # Horizon step 2 (0,1,2) for 15 min (5 min freq)
        'mape_at_15min': MaskedMAPE(at=2),
        'mse_at_15min': MaskedMSE(at=2),
        'mae_at_30min': MaskedMAE(at=5), # Horizon step 5 for 30 min
        'mape_at_30min': MaskedMAPE(at=5),
        'mse_at_30min': MaskedMSE(at=5),
        'mae_at_60min': MaskedMAE(at=11),# Horizon step 11 for 60 min
        'mape_at_60min': MaskedMAPE(at=11),
        'mse_at_60min': MaskedMSE(at=11),
    }

    predictor = Predictor(
        model=stgnn_model,             # Correct model variable
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': 0.001},
        loss_fn=loss_fn,
        metrics=metrics
    )

    logger = TensorBoardLogger(save_dir="logs", name="stgnn_experiment", version="tts_v1") # Changed version name

    # %load_ext tensorboard
    # %tensorboard --logdir logs # These are for Jupyter

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'logs/{logger.name}/{logger.version}/checkpoints', # Save checkpoints in versioned folder
        save_top_k=1,
        monitor='val_mae', # Monitor validation MAE
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=1, # Reduced for quick test, set to 100 or more for real training
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        limit_train_batches=10,  # Reduced for quick test, set to 100 or more
        limit_val_batches=5,     # Added for faster validation during test
        callbacks=[checkpoint_callback]
    )
    
    # --- 6. Train the Model ---
    print("Starting model training...")
    trainer.fit(predictor, datamodule=dm)
    print("Training finished.")

    # --- 7. Test the Model ---
    print("Loading best model and starting testing...")
    # predictor.load_model(checkpoint_callback.best_model_path) # Predictor might load it automatically after fit with ModelCheckpoint
    # The above line might not be needed as PL's test usually uses the best_model_path from checkpoint_callback
    # or the one loaded by the end of fit. Test with and without.
    
    # It's good practice to ensure the model is in eval mode for testing, though Predictor should handle it.
    # predictor.model.eval() # or predictor.eval() if available (Predictor.freeze() does this)
    # predictor.freeze() # This sets model to eval and disables gradients

    test_results = trainer.test(predictor, datamodule=dm, ckpt_path='best')
    print("Testing finished.")
    print("Test results:", test_results)

    # --- 8. Generate Visualizations and Reports ---
    print("\n--- Generating Visualizations and Reports ---")

    # 8.1. Heatmap of Adjacency Matrix
    print("\nGenerating Adjacency Matrix Heatmap...")
    # Retrieve connectivity from the datamodule's dataset or create_datamodule again if simpler
    # For simplicity, let's assume dm.dataset.connectivity holds it if dm is setup with it
    # Or, we can get it from the raw_dataset as in explore_raw_dataset_properties
    if hasattr(dm, 'dataset') and hasattr(dm.dataset, 'connectivity'):
        edge_index, edge_weight = dm.dataset.connectivity
        adj_matrix = edge_index_to_adj(edge_index, edge_weight, n_nodes=dm.n_nodes).cpu().numpy()
        plot_adjacency_heatmap(adj_matrix, title=f"Adjacency Matrix (Threshold {CONNECTIVITY_THRESHOLD})")
    else: # Fallback if connectivity not directly on dm.dataset (depends on TSL version/custom DataModule)
        print("Connectivity not found directly on dm.dataset, re-fetching for heatmap...")
        temp_raw_dataset = MetrLA(root='./data')
        edge_index, edge_weight = temp_raw_dataset.get_connectivity(threshold=CONNECTIVITY_THRESHOLD, layout="edge_index")
        adj_matrix = edge_index_to_adj(edge_index, edge_weight, n_nodes=temp_raw_dataset.n_nodes).cpu().numpy()
        plot_adjacency_heatmap(adj_matrix, title=f"Adjacency Matrix (Threshold {CONNECTIVITY_THRESHOLD})")

    # 8.2. Average Performance Table and Graph
    print("\nGenerating Average Performance Report...")
    horizons_minutes = [15, 30, 60]
    metrics_to_plot = ['mae', 'mape', 'mse']
    performance_data = {
        'Horizon (min)': horizons_minutes,
        'MAE': [test_results[0].get(f'test_mae_at_{h}min', np.nan) for h in horizons_minutes],
        'MAPE': [test_results[0].get(f'test_mape_at_{h}min', np.nan) for h in horizons_minutes],
        'MSE': [test_results[0].get(f'test_mse_at_{h}min', np.nan) for h in horizons_minutes],
    }
    performance_df = pd.DataFrame(performance_data).set_index('Horizon (min)')
    print("\nAverage Performance Metrics Across All Sensors:")
    print(performance_df)
    plot_performance_metrics(performance_df, title="TTS: Avg. Performance vs. Prediction Horizon")

    # 8.3. Per-Station Analysis (Actual vs. Predicted)
    print("\nGenerating Per-Station Analysis Plots (Actual vs. Predicted)...")
    # Get predictions for the test set
    # trainer.predict returns a list of list of batches. We need to concatenate.
    predictions_batched = trainer.predict(predictor, datamodule=dm, ckpt_path='best')
    
    all_predictions = []
    for batch_list in predictions_batched: # trainer.predict may return list of lists if multiple dataloaders
        for batch_preds in batch_list:     # Each element is a batch of predictions
            all_predictions.append(batch_preds.cpu().numpy()) # Prediction tensor from model
    
    if not all_predictions:
        print("No predictions were generated from trainer.predict. Skipping per-station plots.")
    else:
        predictions_np = np.concatenate(all_predictions, axis=0) # Shape: (total_samples, horizon, nodes, features)

        # Get actual values from the test dataloader
        all_actuals = []
        for batch in dm.test_dataloader():
            all_actuals.append(batch.y.cpu().numpy()) # y is usually (batch_size, horizon, nodes, features)
        actuals_np = np.concatenate(all_actuals, axis=0)

        # Ensure shapes match (they should if batching is consistent)
        # The number of samples might differ slightly if drop_last=True/False in dataloaders.
        # We'll plot for the minimum number of samples available in both.
        num_test_samples = min(predictions_np.shape[0], actuals_np.shape[0])
        predictions_np = predictions_np[:num_test_samples]
        actuals_np = actuals_np[:num_test_samples]

        print(f"Predictions shape: {predictions_np.shape}, Actuals shape: {actuals_np.shape}")

        # Plot for the first 3 sensors and specific horizons
        # Horizons for plotting (15min, 30min, 60min correspond to steps 2, 5, 11)
        horizon_steps_to_plot = {
            '15min': 2,
            '30min': 5,
            '60min': 11
        }
        num_sensors_to_plot = 3
        plot_duration = 200 # Number of time steps to show in the plot

        for sensor_i in range(min(num_sensors_to_plot, dm.n_nodes)):
            for horizon_label, horizon_step in horizon_steps_to_plot.items():
                if horizon_step < actuals_np.shape[1]: # Ensure horizon_step is valid
                    plot_actual_vs_predicted(actuals_np, predictions_np, 
                                             sensor_idx=sensor_i, 
                                             horizon_step=horizon_step, 
                                             num_steps=plot_duration, 
                                             title_prefix=f"TTS: Sensor {sensor_i}, Horizon {horizon_label}")
                else:
                    print(f"Skipping plot for sensor {sensor_i}, horizon step {horizon_step} as it exceeds prediction horizon length.")

    print("\n--- Visualizations and Reports Generation Finished ---")

if __name__ == "__main__":
    main()