print("FM_TORCH FINAL VERSION 2", flush=True)
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import os, time, copy
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\vanqu\OneDrive\Desktop\fm_replication\data"
SAVE_DIR = r"C:\Users\vanqu\OneDrive\Desktop\fm_replication"
torch.manual_seed(42)
np.random.seed(42)

# ── Load & convert ───────────────────────────
def load_data(data_dir):
    print("Loading...")
    X_train = sp.load_npz(os.path.join(data_dir,"X_train.npz"))
    X_val   = sp.load_npz(os.path.join(data_dir,"X_val.npz"))
    X_test  = sp.load_npz(os.path.join(data_dir,"X_test.npz"))
    y_train = np.load(os.path.join(data_dir,"y_train.npy"))
    y_val   = np.load(os.path.join(data_dir,"y_val.npy"))
    y_test  = np.load(os.path.join(data_dir,"y_test.npy"))

    def to_tensor(X_sp):
        n   = X_sp.shape[0]
        out = np.zeros((n,2), dtype=np.int64)
        csr = X_sp.tocsr()
        for i in range(n):
            c = csr.indices[csr.indptr[i]:csr.indptr[i+1]]
            out[i,:len(c)] = c
        return torch.tensor(out, dtype=torch.long)

    print("  Converting splits...")
    Xt = to_tensor(X_train)
    Xv = to_tensor(X_val)
    Xe = to_tensor(X_test)
    yt = torch.tensor(y_train, dtype=torch.float32)
    yv = torch.tensor(y_val,   dtype=torch.float32)
    ye = torch.tensor(y_test,  dtype=torch.float32)
    nf = X_train.shape[1]

    print(f"  Train:{Xt.shape} Val:{Xv.shape} Test:{Xe.shape}")
    print(f"  n_features:{nf}  y_mean:{yt.mean():.4f}")
    print(f"  Sample X[0]:{Xt[0].tolist()}  y[0]:{yt[0].item()}")
    return Xt, Xv, Xe, yt, yv, ye, nf


# ── FM Model ─────────────────────────────────
class FM(nn.Module):
    def __init__(self, nf, k=10):
        super().__init__()
        self.k  = k
        self.nf = nf

        # Global bias
        self.w0 = nn.Parameter(torch.zeros(1))

        # Linear weights
        self.w  = nn.Parameter(torch.zeros(nf))

        # Latent factors
        self.V  = nn.Parameter(
            torch.normal(0, 0.01, size=(nf, k))
        )

    def forward(self, idx):
        u     = idx[:, 0]
        m     = idx[:, 1]
        bias  = self.w0.expand(idx.shape[0])
        lin   = self.w[u] + self.w[m]
        inter = (self.V[u] * self.V[m]).sum(1)
        return bias + lin + inter


# ── Evaluate (clamp only here) ───────────────
def evaluate(model, X, y, bs=1024):
    model.eval()
    all_p = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            p = model(X[s:s+bs]).clamp(1, 5)
            all_p.append(p)
    p    = torch.cat(all_p)
    rmse = ((y - p)**2).mean().sqrt().item()
    mae  = (y - p).abs().mean().item()
    return rmse, mae, p


# ── Train ────────────────────────────────────
def train(model, Xt, yt, Xv, yv,
          epochs=50, lr=0.01,
          wd=0.001, bs=512):

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, factor=0.5, min_lr=1e-5
    )
    crit     = nn.MSELoss()
    best_val = 999.
    best_ep  = 1
    best_st  = None
    losses_tr = []
    losses_vl = []

    print(f"\nTraining FM k={model.k} | "
          f"n={len(Xt):,} | lr={lr} | wd={wd}")
    print(f"{'Epoch':>6}|{'TrainRMSE':>11}|"
          f"{'ValRMSE':>9}|{'Time':>6}")
    print("-"*38)

    for ep in range(1, epochs+1):
        t0    = time.time()
        model.train()
        tloss = 0.0
        nt    = 0

        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), bs):
            idx = perm[s:s+bs]
            xb, yb = Xt[idx], yt[idx]

            opt.zero_grad()

            # ── KEY FIX: NO clamp during training ──
            # Clamp kills gradients when predictions
            # are outside [1,5] at initialization.
            # Let the model freely predict, only
            # clamp at evaluation time.
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

            tloss += loss.item() * len(yb)
            nt    += len(yb)

        tr_rmse       = np.sqrt(tloss / nt)
        vl_rmse, _, _ = evaluate(model, Xv, yv)
        losses_tr.append(tr_rmse)
        losses_vl.append(vl_rmse)

        scheduler.step(vl_rmse)

        if vl_rmse < best_val:
            best_val = vl_rmse
            best_ep  = ep
            best_st  = copy.deepcopy(model.state_dict())

        print(f"{ep:>6}|{tr_rmse:>11.4f}|"
              f"{vl_rmse:>9.4f}|{time.time()-t0:>5.1f}s")

    print(f"\nBest Val RMSE: {best_val:.4f} (ep {best_ep})")
    model.load_state_dict(best_st)
    return losses_tr, losses_vl, best_val


# ── RMSE vs k ────────────────────────────────
def rmse_vs_k(Xt, yt, Xv, yv, Xe, ye, nf,
              ks=[2, 5, 10, 20, 50]):
    print(f"\n{'='*40}")
    print("  RMSE vs k  (Paper Figure 2)")
    print(f"{'='*40}")
    res = {}
    for k in ks:
        print(f"\n  --- k={k} ---")
        m = FM(nf, k)
        _, _, bv = train(
            m, Xt, yt, Xv, yv,
            epochs=30, lr=0.01, wd=0.001
        )
        te, _, _ = evaluate(m, Xe, ye)
        res[k]   = {"val": bv, "test": te}
        print(f"  k={k:>3} | val={bv:.4f} | test={te:.4f}")
    return res


# ── Plot ─────────────────────────────────────
def plot(trl, vll, kres, base,
         te_rmse, km, yte, ypred):

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "FM Replication — Rendle 2010 (500K Netflix)",
        fontsize=13, fontweight="bold"
    )

    # Plot 1: Training curve
    a  = ax[0, 0]
    ep = range(1, len(trl)+1)
    a.plot(ep, trl, "b-o", ms=4, label="Train RMSE")
    a.plot(ep, vll, "r-s", ms=4, label="Val RMSE")
    a.axhline(min(vll), color="g", ls="--", alpha=0.7,
              label=f"Best: {min(vll):.4f}")
    a.set(xlabel="Epoch", ylabel="RMSE",
          title=f"Training Curve (k={km})")
    a.legend()
    a.grid(alpha=0.3)

    # Plot 2: RMSE vs k
    a  = ax[0, 1]
    ks = list(kres.keys())
    tr = [kres[k]["test"] for k in ks]
    sv = [base["Linear"]] * len(ks)
    a.plot(ks, tr, "b-o", ms=7, lw=2,
           label="FM (ours)")
    a.plot(ks, sv, "r--^", ms=7, lw=2,
           label="Linear baseline", alpha=0.8)
    for k, r in zip(ks, tr):
        a.annotate(
            f"{r:.3f}", (k, r),
            textcoords="offset points",
            xytext=(0, 8), ha="center", fontsize=8
        )
    a.set(xlabel="Dimensionality k", ylabel="RMSE",
          title="RMSE vs k\n(Replicates Paper Figure 2)")
    a.legend()
    a.grid(alpha=0.3)

    # Plot 3: Model comparison
    a  = ax[1, 0]
    md = {
        "Global\nMean":        base["Mean"],
        "Linear\nRidge":       base["Linear"],
        "FM Scratch\nk=10":    0.9730,
        f"FM PyTorch\nk={km}": te_rmse
    }
    cl  = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]
    bs2 = a.bar(md.keys(), md.values(),
                color=cl, edgecolor="k", lw=0.8)
    for b, v in zip(bs2, md.values()):
        a.text(
            b.get_x() + b.get_width()/2,
            b.get_height() + 0.01,
            f"{v:.4f}", ha="center",
            fontsize=9, fontweight="bold"
        )
    a.set(ylabel="RMSE (lower is better)",
          title="Model Comparison")
    a.set_ylim(0, max(md.values()) * 1.2)
    a.grid(True, axis="y", alpha=0.3)

    # Plot 4: Actual vs Predicted
    a = ax[1, 1]
    a.hist(yte.numpy(), bins=5, alpha=0.6,
           color="steelblue", label="Actual",
           range=(0.5, 5.5), edgecolor="k")
    a.hist(ypred.numpy(), bins=30, alpha=0.6,
           color="orange", label="Predicted",
           range=(0.5, 5.5), edgecolor="k")
    a.set(xlabel="Rating", ylabel="Count",
          title="Actual vs Predicted Distribution")
    a.legend()
    a.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "fm_final_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out}")
    plt.show()


# ── MAIN ─────────────────────────────────────
if __name__ == "__main__":

    # 1. Load
    Xt, Xv, Xe, yt, yv, ye, nf = load_data(DATA_DIR)

    # 2. Train k=10
    KM = 10
    m  = FM(nf, k=KM)
    trl, vll, _ = train(
        m, Xt, yt, Xv, yv,
        epochs=50, lr=0.01, wd=0.001
    )

    # 3. Test evaluation
    te, ma, yp = evaluate(m, Xe, ye)
    print(f"\nTest RMSE : {te:.4f}")
    print(f"Test MAE  : {ma:.4f}")

    # 4. RMSE vs k
    kres = rmse_vs_k(
        Xt, yt, Xv, yv, Xe, ye, nf,
        ks=[2, 5, 10, 20, 50]
    )

    # 5. Baselines
    base = {"Mean": 1.1683, "Linear": 0.9704}

    # 6. Summary
    print(f"\n{'='*40}")
    print("  FINAL RESULTS")
    print(f"{'='*40}")
    print(f"  Global Mean      : 1.1683")
    print(f"  Linear (Ridge)   : 0.9704")
    print(f"  FM Scratch k=10  : 0.9730")
    print(f"  FM PyTorch k={KM}  : {te:.4f}")
    print(f"  Paper (100M)     : ~0.90-0.92")
    print(f"\n  k results:")
    bk = min(v["test"] for v in kres.values())
    for k, r in kres.items():
        tag = " <best" if r["test"] == bk else ""
        print(f"    k={k:>3}: {r['test']:.4f}{tag}")

    # 7. Plot
    plot(trl, vll, kres, base, te, KM, ye, yp)
    print("\nDone!")