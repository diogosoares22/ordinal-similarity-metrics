# Scalable and Interpretable Representation Alignment with Ordinal Similarity

This repository contains the implementation of Ordinal Similarity Metrics, specifically the **Triplet Similarity Index (TSI)** and **Quadruplet Similarity Index (QSI)**. These metrics provide a way to measure the alignment between two data representations (e.g., from different neural network layers or models) based on their ordinal relationships rather than absolute distances. This paper was accepted at **ICML 2026**. Preprint (arXiv): replace when available — `https://arxiv.org/abs/XXXX.XXXXX`



## Installation

We recommend using Conda to manage the environment:

```bash
conda create -n ordinal-similarity-metrics python=3.11
conda activate ordinal-similarity-metrics
pip install -e .
```

## Quick Start: Using TSI and QSI

Here is a simple example of how to compute TSI and QSI between two sets of representations:

```python
import numpy as np
from src.data import RepresentationPair
from src.tsi import EfficientTSI
from src.qsi import EfficientQSI

# 1. Prepare your representations (n_samples, n_features)
X = np.random.rand(100, 128)
Y = np.random.rand(100, 128)

# 2. Define distance functions
d_x = lambda x, y: np.linalg.norm(x - y)
d_y = lambda x, y: np.linalg.norm(x - y)

# 3. Create a RepresentationPair container
representations = RepresentationPair(X=X, Y=Y, d_x=d_x, d_y=d_y)

# 4. Compute TSI (Triplet Similarity Index)
# Measures how often (i,j,k) triplets maintain the same relative ordering
tsi_metric = EfficientTSI()
tsi_score = tsi_metric(representations)
print(f"TSI Score: {tsi_score:.4f}")

# 5. Compute QSI (Quadruplet Similarity Index)
# Measures alignment based on (i,j,k,l) quadruplet comparisons
qsi_metric = EfficientQSI()
qsi_score = qsi_metric(representations)
print(f"QSI Score: {qsi_score:.4f}")
```

## Massive Dataset Example

Approximate TSI and QSI only query sampled indices, so `X` and `Y` can be backed by large `.npy` files with shape `n_samples x n_features`. For fixed `n_samples`, the approximation runtime is constant with respect to the total dataset size: increasing the dataset from one million to one billion rows does not increase the number of sampled triplets/quadruplets.

```python
import numpy as np
from torch.utils.data import Dataset

from src.data import RepresentationPair
from src.qsi import ApproxQSI
from src.tsi import ApproxTSI


class RepresentationDataset(Dataset):
    def __init__(self, path):
        self.X = np.load(path, mmap_mode="r")  # Shape: n_samples x n_features

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


X = RepresentationDataset("path/to/X.npy")
Y = RepresentationDataset("path/to/Y.npy")
d = lambda a, b: np.linalg.norm(a - b)

representations = RepresentationPair(X=X, Y=Y, d_x=d, d_y=d)

print(ApproxTSI(n_samples=10000, n_threads=8)(representations))
print(ApproxQSI(n_samples=10000, n_threads=8)(representations))
```

Here `X[i]` and `Y[i]` read only the requested rows from memory-mapped `.npy` files. Memory usage is controlled by the number of sampled comparisons and the size of each fetched representation, not by the total number of rows in the dataset.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{soares2026tsi,
  author    = {Soares, Diogo and Gawade, Pankhil and Dittadi, Andrea and Szczurek, Ewa},
  title     = {Scalable and Interpretable Representation Alignment with Ordinal Similarity},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  publisher = {PMLR}
}
```