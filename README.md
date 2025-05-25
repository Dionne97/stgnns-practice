# stgnns-practice
Spatio Temporal Graph Neural Network Practise

---

# Spatiotemporal GNN Environment Setup (with TSL)

This README documents the steps to set up a Python environment for running spatiotemporal graph neural network experiments using the [TSL](https://github.com/TorchSpatiotemporal/tsl) library, along with PyTorch and related dependencies, on macOS.

---

## 1. Prerequisites

- **[Miniforge](https://github.com/conda-forge/miniforge) or Anaconda/Miniconda** installed
- **Homebrew** (for system libraries, if needed):  
  Install from [brew.sh](https://brew.sh/) if not already present.

---

## 2. Create and Activate a Conda Environment

```bash
conda create -n stgnn-env python=3.9 -y
conda activate stgnn-env
```

---

## 3. Install Core Dependencies with Conda

Install scientific libraries and HDF5 support (to avoid binary/library conflicts):

```bash
conda install -c conda-forge pytorch pytorch-lightning numpy pandas scikit-learn hdf5 pytables
```

---

## 4. Install PyG and TSL with pip

Some packages (like PyG and TSL) are best installed from source:

```bash
pip install torch-scatter torch-sparse
pip install git+https://github.com/pyg-team/pytorch_geometric.git
pip install git+https://github.com/TorchSpatiotemporal/tsl.git
```

---

## 5. Handling HDF5 and PyTables Issues

If you encounter errors related to HDF5 or PyTables (e.g., `symbol not found in flat namespace '_H5E_CALLBACK_g'`):

- **Uninstall broken pip packages:**
  ```bash
  pip uninstall tables h5py -y
  ```
- **Reinstall with conda:**
  ```bash
  conda install -c conda-forge pytables hdf5
  ```

---

## 6. Handling OpenMP Errors on macOS

If you see errors like:
```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```
You can work around this by running:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python your_script.py
```
> **Note:** This is a workaround. For a robust solution, try to install all scientific packages with conda and avoid mixing pip/Homebrew/conda for these libraries.

---

## 7. Suppressing FutureWarnings

If you see warnings like:
```
FutureWarning: 'T' is deprecated and will be removed in a future version, please use 'min' instead.
```
You can suppress them in your script with:
```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```

---

## 8. Running Your Script

After setup, run your script as usual:
```bash
python stgnns.py
```

---

## 9. Deleting the Conda Environment

To remove the environment if needed:
```bash
conda env remove -n stgnn-env
```

---

## 10. Additional Tips

- List all conda environments:
  ```bash
  conda env list
  ```
- If you need to install system libraries (rare, with conda), use Homebrew:
  ```bash
  brew install hdf5 libomp
  ```

---

## References

- [TSL GitHub](https://github.com/TorchSpatiotemporal/tsl)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [PyTables](https://www.pytables.org/)
- [Conda Documentation](https://docs.conda.io/projects/conda/en/latest/index.html)
