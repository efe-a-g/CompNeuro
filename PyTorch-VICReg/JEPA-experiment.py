# experiment_runner_vicreg.py
import json
from datetime import datetime
import numpy as np
import torch

# Import the VICReg experiment function you added evaluation to.
# Make sure this module name matches your file: bss_vicreg_with_matching.py
from bss_vicreg_with_matching import run_experiment

def run_multiple_vicreg_trials(
        n_trials: int = 10,
        d: int = 2,
        m: int = 2,
        n_samples: int = 20000,
        batch_size: int = 5000,
        lr: float = 1e-2,
        n_epochs: int = 50,
        mu: float = 1.0,
        nu: float = 2.0,
        gamma: float = 1.0,
        device: str = 'cpu',
        save_results: bool = True,
        results_prefix: str = "vicreg_results"
    ):
    """
    Run multiple VICReg-style BSS experiments with a new random mixing matrix A for each trial.

    Returns:
        results: list of dicts, one per trial
    """
    results = []

    for trial in range(1, n_trials + 1):
        print("\n" + "="*60)
        print(f" TRIAL {trial}/{n_trials} (VICReg)")
        print("="*60)

        # Use different seed per trial so A and S differ
        seed = trial * 12345

        model, A, S, X, transformed, epoch_metrics = run_experiment(
            seed=seed,
            d=d,
            m=m,
            n_samples=n_samples,
            batch_size=batch_size,
            lr=lr,
            n_epochs=n_epochs,
            mu=mu,
            nu=nu,
            gamma=gamma,
            device=device
        )

        # final epoch metrics (last entry)
        final_metrics = epoch_metrics[-1]

        # build trial record, converting tensors -> lists for JSON
        trial_record = {
            "trial": trial,
            "seed": int(seed),
            "d": int(d),
            "m": int(m),
            "n_samples": int(n_samples),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "n_epochs": int(n_epochs),
            "A": A.detach().cpu().numpy().tolist(),
            "W": model.fc.weight.detach().cpu().numpy().tolist(),
            "A_times_W": (A @ model.fc.weight.detach()).cpu().numpy().tolist(),
            "final_perm": final_metrics.get("perm"),
            "mean_abs_corr": float(final_metrics.get("mean_abs_corr")),
            "mse_matched": float(final_metrics.get("mse_matched")),
            "metrics_per_epoch": epoch_metrics
        }

        results.append(trial_record)

        print(f"  --> mean_abs_corr = {trial_record['mean_abs_corr']:.6f}, mse_matched = {trial_record['mse_matched']:.6f}")
        print(f"  --> final_perm = {trial_record['final_perm']}")

    # Save results to JSON if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"{results_prefix}_vicreg_{timestamp}.json"
        # ensure numpy types are converted (they should be since we converted arrays to lists)
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to: {out_file}")

    return results


if __name__ == "__main__":
    # Example usage: run 20 trials
    results = run_multiple_vicreg_trials(
        n_trials=20,
        d=2,
        m=2,
        n_samples=20000,
        batch_size=1024,
        lr=1e-2,
        n_epochs=50,
        mu=1.0,
        nu=1.0,
        gamma=1.0,
        device='cpu',
        save_results=True,
        results_prefix="vicreg_expts"
    )

    # Summary aggregation
    mean_corrs = np.array([r["mean_abs_corr"] for r in results])
    mse_vals   = np.array([r["mse_matched"] for r in results])
    print("\n=== AGGREGATED SUMMARY ===")
    print(f"Trials: {len(results)}")
    print(f"Mean(mean_abs_corr) = {mean_corrs.mean():.6f}, std = {mean_corrs.std():.6f}")
    print(f"Mean(mse_matched)   = {mse_vals.mean():.6e}, std = {mse_vals.std():.6e}")
