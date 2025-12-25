# Inspired by VICReg applied to JEPA

import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Variance and Covariance Losses
def variance_loss(Z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(Z.var(dim=0, unbiased=True) + eps)
    hinge = torch.relu(gamma - std)
    return hinge.mean()

def covariance_loss(Z: torch.Tensor) -> torch.Tensor:
    n, d = Z.shape
    if n <= 1:
        return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)
    Zc = Z - Z.mean(dim=0, keepdim=True)
    cov = (Zc.T @ Zc) / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    return off.pow(2).sum() / float(d)

def var_cov_loss(Z: torch.Tensor, mu: float = 25.0, nu: float = 1.0,
                 gamma: float = 1.0, eps: float = 1e-4):
    v = variance_loss(Z, gamma=gamma, eps=eps)
    c = covariance_loss(Z)
    return mu * v + nu * c, v, c

# Activation Function in Second Layer to Clip outputs
class ClipActivation(nn.Module):
    def __init__(self, a=math.sqrt(3)):
        super().__init__()
        self.a = a

    def forward(self, x):
        return torch.clamp(x, -self.a, self.a)

# Two layer network with activation
class LinearUnmix(nn.Module):
    def __init__(self, m: int, d: int, bias: bool = False):
        super().__init__()
        self.fc = nn.Linear(m, d, bias=bias)
        self.clip = ClipActivation(a=math.sqrt(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc(x)
        return self.clip(z)

# Generate data
def sample_sources(n_samples: int, d: int, device='cpu'):
    a = math.sqrt(3.0)
    return (2 * a) * torch.rand(n_samples, d, device=device) - a

def sample_mixing_matrix(m: int, d: int, device='cpu'):
    return torch.randn(m, d, device=device)

# Matching
def greedy_match_abs_correlation(S: torch.Tensor, Z: torch.Tensor):
    """
    Greedy matching based on absolute correlations.
    Returns:
      perm: list of length d, where perm[i] = index of Z-column matched to S-column i
      mean_abs_corr: mean absolute correlation across matched pairs
      mse_matched: MSE after aligning Z columns according to perm (Z_permuted -> S)
      corr_matrix: (d,d) matrix of absolute correlations used for matching
    Notes:
      - S and Z should be same shape (N, d).
      - Works on the device of input tensors.
      - Greedy matching isn't globally optimal but is deterministic and avoids external deps.
    """
    assert S.shape[1] == Z.shape[1], "S and Z must have same number of columns (d)"
    N, d = S.shape
    # center & normalize columns
    S_center = S - S.mean(dim=0, keepdim=True)
    Z_center = Z - Z.mean(dim=0, keepdim=True)
    # avoid division by zero
    S_std = S_center.std(dim=0, unbiased=False) + 1e-9
    Z_std = Z_center.std(dim=0, unbiased=False) + 1e-9
    S_norm = S_center / S_std
    Z_norm = Z_center / Z_std
    # correlation matrix (d x d)
    # corr_ij = (S[:,i] dot Z[:,j]) / N
    corr = (S_norm.T @ Z_norm) / float(N)
    corr_abs = corr.abs().clone()

    # Greedy assignment on corr_abs
    perm = [-1] * d
    assigned_rows = torch.zeros(d, dtype=torch.bool, device=S.device)
    assigned_cols = torch.zeros(d, dtype=torch.bool, device=S.device)
    corr_abs_work = corr_abs.clone()
    for _ in range(d):
        # choose max element
        max_val = corr_abs_work.max()
        if max_val <= -0.5:  # sentinel if something went wrong (shouldn't)
            break
        max_idx = corr_abs_work.argmax()
        i = int(max_idx // d)
        j = int(max_idx % d)
        perm[i] = j
        assigned_rows[i] = True
        assigned_cols[j] = True
        # inhibit selected row and column
        corr_abs_work[i, :] = -1.0
        corr_abs_work[:, j] = -1.0

    # If any row left unassigned (shouldn't happen), assign remaining cols arbitrarily
    remaining_rows = [i for i, p in enumerate(perm) if p == -1]
    remaining_cols = [j for j in range(d) if not assigned_cols[j]]
    for i, j in zip(remaining_rows, remaining_cols):
        perm[i] = j

    # compute matched correlation and MSE
    perm_tensor = torch.tensor(perm, dtype=torch.long, device=S.device)
    # create Z permuted with columns permuted to align to S columns
    Z_permuted = Z[:, perm_tensor]
    matched_corrs = (S_norm * (Z_permuted - Z_permuted.mean(0, keepdim=True)) / (Z_permuted.std(dim=0, unbiased=False)+1e-9)).mean(dim=0).abs()
    mean_abs_corr = float(matched_corrs.mean().item())
    mse_matched = float(((Z_permuted - S).pow(2).mean()).item())

    return perm, mean_abs_corr, mse_matched, corr.abs()

# Training experiment
def run_experiment(seed: int = 0,
                   d: int = 2,
                   m: int = 2,
                   n_samples: int = 20000,
                   batch_size: int = 512,
                   lr: float = 1e-3,
                   n_epochs: int = 30,
                   mu: float = 1.0,
                   nu: float = 1.0,
                   gamma: float = 1.0,
                   device: str = 'cpu'):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 1) generate sources and mixtures
    S = sample_sources(n_samples, d, device=device)        # (N, d)
    A = sample_mixing_matrix(m, d, device=device)          # (m, d)
    X = S @ A.T                                           # (N, m) -> x = A s

    ds = TensorDataset(X, S)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = LinearUnmix(m=m, d=d, bias=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training: d={d}, m={m}, N={n_samples}, batch={batch_size}, epochs={n_epochs}")
    epoch_metrics = []
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        total_v = 0.0
        total_c = 0.0
        num_batches = 0
        for xb, sb in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            zb = model(xb)                     # (batch, d)
            loss, v_term, c_term = var_cov_loss(zb, mu=mu, nu=nu, gamma=gamma)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_v += float(v_term.item())
            total_c += float(c_term.item())
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_v = total_v / max(1, num_batches)
        avg_c = total_c / max(1, num_batches)

        # Evaluation on entire dataset: measure recovery quality
        with torch.no_grad():
            W = model.fc.weight  # shape (d, m)
            transformed = X @ W.T  # (N, d)
            perm, mean_abs_corr, mse_matched, corr_matrix = greedy_match_abs_correlation(S, transformed)
            epoch_metrics.append({
                'epoch': epoch,
                'vicreg_loss': avg_loss,
                'v_term': avg_v,
                'c_term': avg_c,
                'mean_abs_corr': mean_abs_corr,
                'mse_matched': mse_matched,
                'perm': perm
            })

        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f}, v={avg_v:.6f}, c={avg_c:.6f}, mean_abs_corr={mean_abs_corr:.6f}, mse_matched={mse_matched:.6f}")

    # final model and metrics
    # Also compute final transformed for return
    with torch.no_grad():
        W = model.fc.weight
        transformed = X @ W.T

    return model, A, S, X, transformed, epoch_metrics

if __name__ == "__main__":
    model, A, S, X, transformed, metrics = run_experiment(seed=100,
                                      d=2,
                                      m=2,
                                      n_samples=20000,
                                      batch_size=5000,
                                      lr=1e-2,
                                      n_epochs=50,
                                      mu=1.0,
                                      nu=2.0,
                                      gamma=1.0,
                                      device='cpu')
    W = model.fc.weight
    print("A @ W =\n", A @ W)
    print("Last epoch metrics:", metrics[-1])

    with torch.no_grad():
        def scatter2d(P,label_):
            plt.plot(P[:, 0].cpu().numpy(), P[:, 1].cpu().numpy(), 'o', linestyle='None', markersize=4,label=label_)
        scatter2d(S, "S")
        scatter2d(X,"X")
        scatter2d(transformed,"approx S")
        plt.legend()
        plt.show()
