import torch
import numpy as np
import scipy.sparse as sp
import os


DATA_DIR = r"C:\Users\vanqu\OneDrive\Desktop\fm_replication\data"

# Load data
X_train = sp.load_npz(os.path.join(DATA_DIR, "X_train.npz"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

# Convert one row manually
csr = X_train.tocsr()
row0 = csr.indices[csr.indptr[0]:csr.indptr[1]]
row1 = csr.indices[csr.indptr[1]:csr.indptr[2]]

print(f"Row 0 indices : {row0}  | y={y_train[0]}")
print(f"Row 1 indices : {row1}  | y={y_train[1]}")
print(f"Row 0 length  : {len(row0)}  (should be 2)")

# Build index tensor for 5 samples
n = 5
out = np.zeros((n, 2), dtype=np.int64)
for i in range(n):
    cols = csr.indices[csr.indptr[i]:csr.indptr[i+1]]
    out[i] = cols
X_tensor = torch.tensor(out, dtype=torch.long)
y_tensor = torch.tensor(y_train[:n], dtype=torch.float32)

print(f"\nX_tensor:\n{X_tensor}")
print(f"y_tensor: {y_tensor}")

# Build tiny FM and do ONE forward pass
n_features = X_train.shape[1]
torch.manual_seed(42)

w0 = torch.zeros(1, requires_grad=True)
w  = torch.zeros(n_features, requires_grad=True)
V  = torch.normal(0, 0.01, size=(n_features, 10),
                  requires_grad=True)

user_idx  = X_tensor[:, 0]
movie_idx = X_tensor[:, 1]

bias    = w0.expand(n)
linear  = w[user_idx] + w[movie_idx]
v_user  = V[user_idx]
v_movie = V[movie_idx]
inter   = (v_user * v_movie).sum(dim=1)
preds   = bias + linear + inter

print(f"\nPredictions before training : {preds.detach()}")
print(f"Targets                     : {y_tensor}")

# One SGD step
loss = ((preds - y_tensor)**2).mean()
loss.backward()
print(f"\nLoss         : {loss.item():.4f}")
print(f"w0 gradient  : {w0.grad.item():.6f}")
print(f"w gradient   : {w.grad[user_idx[0]].item():.6f}")
print(f"V gradient   : {V.grad[user_idx[0], 0].item():.6f}")
print(f"\nGradients flowing: {'YES' if w0.grad.item() != 0 else 'NO'}")