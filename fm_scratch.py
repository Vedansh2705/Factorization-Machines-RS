import numpy as np
import scipy.sparse as sp
import os
import time
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = r"C:\Users\vanqu\OneDrive\Desktop\fm_replication\data"

# ─────────────────────────────────────────────
# STEP 1: Load splits
# ─────────────────────────────────────────────
def load_splits(data_dir):
    print("Loading data splits...")
    X_train = sp.load_npz(os.path.join(data_dir, "X_train.npz"))
    X_val   = sp.load_npz(os.path.join(data_dir, "X_val.npz"))
    X_test  = sp.load_npz(os.path.join(data_dir, "X_test.npz"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_val   = np.load(os.path.join(data_dir, "y_val.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

    print(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}  y_val   : {y_val.shape}")
    print(f"  X_test  : {X_test.shape}  y_test  : {y_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# STEP 2: FM Model Class
#
# Paper equation (1):
#   y(x) = w0
#         + sum_i(wi * xi)
#         + sum_i sum_j>i <vi, vj> xi xj
#
# Parameters:
#   w0        : global bias (scalar)
#   w         : feature weights (n_features,)
#   V         : latent factor matrix (n_features x k)
#
# Key trick for O(kn) computation — paper Lemma 3.1:
#   sum_i sum_j>i <vi,vj> xi xj
#   = 0.5 * sum_f [ (sum_i vi,f * xi)^2
#                  - sum_i (vi,f * xi)^2 ]
# ─────────────────────────────────────────────
class FactorizationMachine:

    def __init__(self, n_features, k=10, learning_rate=0.01,
                 reg_w0=0.01, reg_w=0.01, reg_v=0.01, seed=42):
        """
        n_features  : size of feature vector (|U| + |I|)
        k           : latent factor dimension (hyperparameter)
        learning_rate: step size for SGD
        reg_w0/w/v  : L2 regularization strengths
        """
        np.random.seed(seed)
        self.k             = k
        self.lr            = learning_rate
        self.reg_w0        = reg_w0
        self.reg_w         = reg_w
        self.reg_v         = reg_v
        self.n_features    = n_features

        # ── Initialize parameters ─────────────────
        # w0: scalar bias
        self.w0 = 0.0

        # w: feature weights, initialized to 0
        self.w  = np.zeros(n_features, dtype=np.float64)

        # V: latent factors, initialized with small random values
        # Paper uses normal distribution with std = 0.01
        self.V  = np.random.normal(0, 0.01,
                                   size=(n_features, k))

        # Training history
        self.train_losses = []
        self.val_losses   = []

    # ── Forward pass: compute y_hat for one sample ──
    def predict_one(self, x):
        """
        x: sparse row vector (1 x n_features)
        Returns scalar prediction y_hat

        Implements Paper Equation (1) using Lemma 3.1
        for O(kn) computation.
        """
        # Convert sparse row to dense array
        x_dense = np.asarray(x.todense()).flatten()

        # Term 1: global bias
        term_bias = self.w0

        # Term 2: linear terms  sum_i wi*xi
        term_linear = self.w.dot(x_dense)

        # Term 3: pairwise interactions using Lemma 3.1
        # For each factor f:
        #   (sum_i vi,f * xi)^2 - sum_i (vi,f * xi)^2
        Vx         = self.V * x_dense[:, np.newaxis]   # (n_features x k)
        sum_sq     = np.sum(Vx, axis=0) ** 2            # (k,) squared sum
        sq_sum     = np.sum(Vx ** 2, axis=0)            # (k,) sum of squares
        term_inter = 0.5 * np.sum(sum_sq - sq_sum)      # scalar

        return term_bias + term_linear + term_inter

    # ── Batch predict (for evaluation) ──────────────
    def predict(self, X):
        """
        X: sparse matrix (n_samples x n_features)
        Returns array of predictions (n_samples,)
        """
        # Efficient batch computation using matrix ops
        # Convert to dense for batch (acceptable for eval)
        X_dense = np.asarray(X.todense())               # (n x n_features)

        # Term 1: bias
        bias = np.full(X_dense.shape[0], self.w0)

        # Term 2: linear
        linear = X_dense.dot(self.w)                    # (n,)

        # Term 3: interactions — Lemma 3.1 vectorized
        # XV shape: (n x k)
        XV     = X_dense.dot(self.V)                    # (n x k)
        XV2    = X_dense.dot(self.V ** 2)               # (n x k)
        inter  = 0.5 * np.sum(XV**2 - XV2, axis=1)     # (n,)

        return bias + linear + inter

    # ── Clip predictions to valid rating range ───────
    def clip(self, y_hat):
        return np.clip(y_hat, 1.0, 5.0)

    # ── SGD update for one sample ────────────────────
    def sgd_update(self, x, y):
        """
        Stochastic Gradient Descent update.
        Paper Section III-C: gradient equations (4).

        For squared loss: L = (y_hat - y)^2
        Gradient w.r.t. parameter theta:
            dL/dtheta = 2*(y_hat - y) * dy_hat/dtheta
        """
        x_dense = np.asarray(x.todense()).flatten()
        y_hat   = self.predict_one(x)
        error   = y_hat - y                             # residual

        # Precompute sum_j vj,f * xj for each factor f
        # This is the "precomputed sum" mentioned in paper eq(4)
        # Shape: (k,)
        Vx_sum = self.V.T.dot(x_dense)                 # (k,)

        # ── Gradient updates (Paper Equation 4) ──────

        # dL/dw0 = error * 1
        self.w0 -= self.lr * (error + self.reg_w0 * self.w0)

        # dL/dwi = error * xi
        # Only update non-zero features (sparse efficiency)
        nz = x_dense != 0
        self.w[nz] -= self.lr * (
            error * x_dense[nz] + self.reg_w * self.w[nz]
        )

        # dL/dvi,f = error * xi * (sum_j vj,f*xj - vi,f*xi)
        # Shape update: (n_nonzero x k)
        nz_idx = np.where(nz)[0]
        for i in nz_idx:
            grad_v = error * x_dense[i] * (
                Vx_sum - self.V[i] * x_dense[i]
            )
            self.V[i] -= self.lr * (grad_v + self.reg_v * self.V[i])

        return error ** 2   # return squared error for loss tracking

    # ── Metrics ─────────────────────────────────────
    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    # ── Training loop ────────────────────────────────
    def fit(self, X_train, y_train, X_val, y_val, n_epochs=20):
        """
        Train FM with SGD over n_epochs.
        Evaluates on validation set after each epoch.
        """
        n_samples  = X_train.shape[0]
        n_features = X_train.shape[1]

        print(f"\nTraining Factorization Machine (from scratch)")
        print(f"  Samples    : {n_samples:,}")
        print(f"  Features   : {n_features:,}")
        print(f"  k (factors): {self.k}")
        print(f"  LR         : {self.lr}")
        print(f"  Epochs     : {n_epochs}")
        print(f"  Reg (w,V)  : {self.reg_w}, {self.reg_v}")
        print(f"\n  Epoch |  Train RMSE  |  Val RMSE  |  Time")
        print(f"  {'-'*50}")

        for epoch in range(1, n_epochs + 1):
            start = time.time()

            # Shuffle training data each epoch
            idx = np.random.permutation(n_samples)

            epoch_loss = 0.0
            for i in idx:
                x_i   = X_train[i]
                y_i   = y_train[i]
                sq_err = self.sgd_update(x_i, y_i)
                epoch_loss += sq_err

            train_rmse = np.sqrt(epoch_loss / n_samples)

            # Validation RMSE
            y_val_pred = self.clip(self.predict(X_val))
            val_rmse   = self.rmse(y_val, y_val_pred)

            elapsed = time.time() - start

            self.train_losses.append(train_rmse)
            self.val_losses.append(val_rmse)

            print(f"  {epoch:>5} | {train_rmse:>11.4f}  | {val_rmse:>10.4f} | {elapsed:.1f}s")

        print(f"\n  Training complete!")
        print(f"  Best Val RMSE : {min(self.val_losses):.4f} "
              f"(epoch {np.argmin(self.val_losses)+1})")

        return self


# ─────────────────────────────────────────────
# STEP 3: Baseline Models for comparison
# (Paper Figure 2 — FM vs SVM/Linear)
# ─────────────────────────────────────────────
def train_baselines(X_train, y_train, X_val, y_val):
    from sklearn.linear_model import Ridge
    from sklearn.dummy import DummyRegressor

    print("\nTraining Baseline Models...")
    baselines = {}

    # Baseline 1: Global mean (simplest possible predictor)
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    y_pred = np.clip(dummy.predict(X_val), 1, 5)
    rmse   = np.sqrt(np.mean((y_val - y_pred)**2))
    baselines["Global Mean"] = rmse
    print(f"  Global Mean RMSE  : {rmse:.4f}")

    # Baseline 2: Linear model (= FM with k=0, like linear SVM)
    # This is equivalent to paper's linear SVM (eq. 7 and 10)
    print("  Training Linear model (Ridge)... ", end="", flush=True)
    ridge = Ridge(alpha=1.0)

    # Use sparse matrix directly — Ridge handles it
    ridge.fit(X_train, y_train)
    y_pred = np.clip(ridge.predict(X_val), 1, 5)
    rmse   = np.sqrt(np.mean((y_val - y_pred)**2))
    baselines["Linear (Ridge)"] = rmse
    print(f"Val RMSE: {rmse:.4f}")

    return baselines


# ─────────────────────────────────────────────
# STEP 4: Evaluate on test set
# ─────────────────────────────────────────────
def evaluate(fm, X_test, y_test, baselines):
    print(f"\n{'='*50}")
    print(f"  FINAL TEST SET EVALUATION")
    print(f"{'='*50}")

    y_pred = fm.clip(fm.predict(X_test))
    test_rmse = fm.rmse(y_test, y_pred)
    test_mae  = fm.mae(y_test, y_pred)

    print(f"\n  FM (k={fm.k}):")
    print(f"    RMSE : {test_rmse:.4f}")
    print(f"    MAE  : {test_mae:.4f}")

    print(f"\n  Baselines:")
    for name, rmse in baselines.items():
        diff = rmse - test_rmse
        print(f"    {name:<20}: RMSE {rmse:.4f}  "
              f"(FM better by {diff:+.4f})")

    return test_rmse, test_mae


# ─────────────────────────────────────────────
# STEP 5: Plots
# ─────────────────────────────────────────────
def plot_results(fm, baselines, test_rmse):

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Factorization Machine — Results (Paper Replication)",
                 fontsize=13, fontweight="bold")

    # ── Plot 1: Training curve (loss vs epoch) ───────
    ax = axes[0]
    epochs = range(1, len(fm.train_losses) + 1)
    ax.plot(epochs, fm.train_losses, "b-o", markersize=4,
            label="Train RMSE")
    ax.plot(epochs, fm.val_losses,   "r-s", markersize=4,
            label="Val RMSE")
    ax.axhline(y=min(fm.val_losses), color="green",
               linestyle="--", alpha=0.5,
               label=f"Best Val: {min(fm.val_losses):.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Model Comparison bar chart ───────────
    ax = axes[1]
    all_models = {**baselines, f"FM (k={fm.k})": test_rmse}
    names  = list(all_models.keys())
    values = list(all_models.values())
    colors = ["#e74c3c" if "FM" not in n else "#2ecc71" for n in names]
    bars   = ax.bar(names, values, color=colors, edgecolor="black",
                    linewidth=0.8, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Model Comparison\n(Replicates Paper Figure 2)")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    # ── Plot 3: Prediction distribution ──────────────
    ax = axes[2]
    X_test_data = sp.load_npz(
        os.path.join(DATA_DIR, "X_test.npz"))
    y_test_data = np.load(
        os.path.join(DATA_DIR, "y_test.npy"))
    y_pred = fm.clip(fm.predict(X_test_data))
    ax.hist(y_test_data, bins=5, alpha=0.6,
            label="Actual", color="steelblue",
            range=(0.5, 5.5), edgecolor="black")
    ax.hist(y_pred, bins=20, alpha=0.6,
            label="Predicted", color="orange",
            range=(0.5, 5.5), edgecolor="black")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Actual vs Predicted\nRating Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(
        r"C:\Users\vanqu\OneDrive\Desktop\fm_replication",
        "fm_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(DATA_DIR)
    n_features = X_train.shape[1]

    # 2. Train FM from scratch
    fm = FactorizationMachine(
        n_features    = n_features,
        k             = 10,       # latent factors (try 10, 20, 50)
        learning_rate = 0.01,
        reg_w0        = 0.01,
        reg_w         = 0.01,
        reg_v         = 0.01
    )
    fm.fit(X_train, y_train, X_val, y_val, n_epochs=20)

    # 3. Train baselines
    baselines = train_baselines(X_train, y_train, X_val, y_val)

    # 4. Final evaluation
    test_rmse, test_mae = evaluate(fm, X_test, y_test, baselines)

    # 5. Plot everything
    plot_results(fm, baselines, test_rmse)

    print(f"\nDone! FM from scratch complete.")
    print(f"Next: Step 5 — PyTorch FM (faster + GPU ready)")