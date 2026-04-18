import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.append("/Users/efe/Desktop/copy/CorInfoMaxCodes/src")

from bss_utils import generate_uncorrelated_uniform_sources, addWGN
from PredictiveDecorrBSS import PredictiveDecorrBSS
from PredictiveDecorrBSSSimple import PredictiveDecorrBSSSimple


def run_single(seed, N=100000, NumberofSources=2, SNR=30):
    np.random.seed(seed)

    NumberofMixtures = NumberofSources + 2

    S = generate_uncorrelated_uniform_sources(
        NumberofSources, N, min_val=-1, max_val=1
    )

    A = np.random.randn(NumberofMixtures, NumberofSources)
    X_noNoise = A @ S
    X = addWGN(X_noNoise, SNR)

    hyperparam_dict = {
        "n_sources": NumberofSources,
        "presumed_domain": "antisparse",
        "lambda_lateral": 0.99,
        "gamma_predictive": 100,
        "lr_W": 1e-1,
        "neural_lr_start": 0.5,
        "neural_lr_stop": 1e-6,
        "neural_dynamics_iterations": 250,
        "neural_OUTPUT_COMP_TOL": 1e-7,
        "lr_W_rule": "divide_by_log_index",
        "lr_W_decay_divider": 5000,
        "neural_lr_rule": "divide_by_loop_index",
        "neural_lr_decay_divider": 200,
        "W": None,
        "C_y": None,
        "mu_y": None,
        "Sgt": S,
        "debug_iteration_point": 10000,
        "plot_debug_during_training": False,
    }

    model = PredictiveDecorrBSSSimple(**hyperparam_dict)
    model.fit(X)

    Y_ = model.predict(X)
    Y_ = model.signed_and_permutation_corrected_sources(S, Y_)

    coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
    Y_ = coef_ * Y_

    snr_vals = np.array(model.ComputeSNR(S, Y_))  # shape: (n_sources,)
    return snr_vals


# -----------------------
# Run experiments
# -----------------------
num_runs = 10
NumberofSources = 2

all_snrs = np.zeros((num_runs, NumberofSources))

base_seed = np.random.randint(5_000_000)
print("Base seed:", base_seed)

for i in tqdm(range(num_runs)):
    all_snrs[i] = run_single(base_seed + i, NumberofSources=NumberofSources)

# -----------------------
# Final statistics
# -----------------------
mean_snr_per_channel = np.mean(all_snrs, axis=0)
std_snr_per_channel = np.std(all_snrs, axis=0, ddof=1)

print("\n=== Per-channel results ===")
for ch in range(NumberofSources):
    print(
        f"Channel {ch}: mean SNR = {mean_snr_per_channel[ch]:.4f}, "
        f"std = {std_snr_per_channel[ch]:.4f}"
    )