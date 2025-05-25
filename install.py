import os
import torch
import subprocess

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

def pip_install(cmd):
    subprocess.check_call(cmd, shell=True)

pip_install(f"pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html")
pip_install(f"pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html")
pip_install("pip install git+https://github.com/pyg-team/pytorch_geometric.git")
pip_install("pip install git+https://github.com/TorchSpatiotemporal/tsl.git")