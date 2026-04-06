import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = r"C:\Users\vanqu\OneDrive\Desktop\fm_replication\data"
RATINGS_CSV = os.path.join(DATA_DIR, "ratings_sample.csv")

# ─────────────────────────────────────────────
# STEP 1: Load cleaned ratings
# ─────────────────────────────────────────────
def load_data(filepath):
    print("Loading cleaned ratings...")
    df = pd.read_csv(filepath)
    print(f"  Shape        : {df.shape}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"  Sample rows  :")
    print(df[["user_idx", "movie_idx", "rating"]].head(3).to_string(index=False))
    return df

# ─────────────────────────────────────────────
# STEP 2: Build sparse feature matrix
#
# Paper Figure 1 explanation:
#   Each row = one (user, movie) interaction
#   Feature vector = [user_onehot | movie_onehot]
#
#   If user_idx = 3 and movie_idx = 7, and |U|=10, |I|=5:
#   x = [0,0,0,1,0,0,0,0,0,0 | 0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
#         ← user block (10) →   ← movie block (|I|) →
#
# We use scipy.sparse (CSR format) because:
#   - Full matrix would be 215K × 215K = 46 BILLION cells
#   - Sparse stores only non-zero values (just 2 per row!)
# ─────────────────────────────────────────────
def build_feature_matrix(df):
    print("\nBuilding sparse feature matrix (Paper Figure 1 style)...")

    n_users  = df["user_idx"].max() + 1
    n_movies = df["movie_idx"].max() + 1
    n_samples = len(df)
    n_features = n_users + n_movies   # total feature vector size

    print(f"  n_users    |U| : {n_users:,}")
    print(f"  n_movies   |I| : {n_movies:,}")
    print(f"  n_features     : {n_features:,}  (= |U| + |I|)")
    print(f"  n_samples      : {n_samples:,}")
    print(f"  Non-zeros/row  : 2  (one user + one movie indicator)")

    # ── Build COO sparse matrix ──────────────────
    # For each sample i:
    #   col for user  = user_idx
    #   col for movie = n_users + movie_idx
    # Each sample contributes exactly 2 non-zero entries

    row_indices  = np.repeat(np.arange(n_samples), 2)   # [0,0, 1,1, 2,2, ...]

    user_cols    = df["user_idx"].values                 # user column indices
    movie_cols   = n_users + df["movie_idx"].values      # movie column indices (offset by |U|)
    col_indices  = np.concatenate([
        user_cols.reshape(-1,1),
        movie_cols.reshape(-1,1)
    ], axis=1).flatten()
    # col_indices now interleaves: [u0, m0, u1, m1, u2, m2, ...]
    # We need to re-order to match row_indices pattern
    col_indices = np.empty(2 * n_samples, dtype=np.int32)
    col_indices[0::2] = user_cols    # even positions = user cols
    col_indices[1::2] = movie_cols   # odd positions  = movie cols

    data = np.ones(2 * n_samples, dtype=np.float32)     # all values = 1.0

    X = sp.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_samples, n_features)
    )

    y = df["rating"].values.astype(np.float32)

    print(f"\n  Sparse matrix X shape   : {X.shape}")
    print(f"  Sparse matrix nnz       : {X.nnz:,}  (non-zero elements)")
    print(f"  Memory (sparse)         : ~{X.data.nbytes / 1024:.1f} KB")
    dense_mb = (n_samples * n_features * 4) / (1024**2)
    print(f"  Memory (if dense)       : ~{dense_mb:,.0f} MB  (why we use sparse!)")
    print(f"  Target vector y shape   : {y.shape}")
    print(f"  Target range            : {y.min():.0f} – {y.max():.0f}")

    return X, y, n_users, n_movies


# ─────────────────────────────────────────────
# STEP 3: Verify one sample (sanity check)
# Paper Example: Alice rates Titanic
# ─────────────────────────────────────────────
def verify_feature_vector(X, df, sample_idx=0):
    print(f"\nVerifying feature vector for sample {sample_idx}:")

    row  = X[sample_idx]                     # sparse row
    cols = row.indices                        # non-zero column positions
    vals = row.data                           # non-zero values

    user_idx  = df["user_idx"].iloc[sample_idx]
    movie_idx = df["movie_idx"].iloc[sample_idx]
    rating    = df["rating"].iloc[sample_idx]
    n_users   = df["user_idx"].max() + 1

    print(f"  user_idx   : {user_idx}")
    print(f"  movie_idx  : {movie_idx}")
    print(f"  rating     : {rating}")
    print(f"  Non-zero cols in x : {cols}  (values: {vals})")
    print(f"  Expected cols      : [{user_idx}, {n_users + movie_idx}]")

    # Confirm structure
    assert user_idx in cols, "ERROR: user column missing!"
    assert (n_users + movie_idx) in cols, "ERROR: movie column missing!"
    assert len(cols) == 2, "ERROR: should have exactly 2 non-zeros!"
    print(f"  Sanity check PASSED — exactly 2 non-zeros, correct positions!")


# ─────────────────────────────────────────────
# STEP 4: Train/Test split
# ─────────────────────────────────────────────
def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    print(f"\nSplitting data...")

    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: from 80%, take 10% as validation
    val_ratio = val_size / (1 - test_size)   # = 0.1/0.8 = 0.125
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    print(f"  Train  : {X_train.shape[0]:,} samples  ({X_train.shape[0]/len(y)*100:.1f}%)")
    print(f"  Val    : {X_val.shape[0]:,} samples  ({X_val.shape[0]/len(y)*100:.1f}%)")
    print(f"  Test   : {X_test.shape[0]:,} samples  ({X_test.shape[0]/len(y)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# STEP 5: Save everything for next steps
# ─────────────────────────────────────────────
def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, data_dir):
    print(f"\nSaving splits to disk...")
    sp.save_npz(os.path.join(data_dir, "X_train.npz"), X_train)
    sp.save_npz(os.path.join(data_dir, "X_val.npz"),   X_val)
    sp.save_npz(os.path.join(data_dir, "X_test.npz"),  X_test)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(data_dir, "y_test.npy"),  y_test)
    print(f"  Saved: X_train, X_val, X_test (sparse .npz)")
    print(f"  Saved: y_train, y_val, y_test (.npy)")
    print(f"  Location: {data_dir}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load
    df = load_data(RATINGS_CSV)

    # 2. Build feature matrix
    X, y, n_users, n_movies = build_feature_matrix(df)

    # 3. Verify one sample
    verify_feature_vector(X, df, sample_idx=0)

    # 4. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 5. Save
    save_splits(X_train, X_val, X_test,
                y_train, y_val, y_test, DATA_DIR)

    print(f"\n{'='*50}")
    print(f"  FEATURE ENGINEERING COMPLETE!")
    print(f"{'='*50}")
    print(f"  X_train shape : {X_train.shape}")
    print(f"  Each row has  : exactly 2 non-zeros")
    print(f"  Format        : [user_onehot | movie_onehot]")
    print(f"  This matches  : Paper Figure 1 exactly!")
    print(f"\n  Ready for Step 4: FM Model Implementation!")