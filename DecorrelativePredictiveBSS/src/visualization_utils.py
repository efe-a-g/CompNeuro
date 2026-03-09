import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Math, display

########### LATEX Style Display Matrix ###############
def display_matrix(array):
    """Display given numpy array with Latex format in Jupyter Notebook.

    Args:
        array (numpy array): Array to be displayed
    """
    data = ""
    for line in array:
        if len(line) == 1:
            data += " %.3f &" % line + r" \\\n"
            continue
        for element in line:
            data += " %.3f &" % element
        data += r" \\" + "\n"
    display(Math("\\begin{bmatrix} \n%s\\end{bmatrix}" % data))

def plot_bss_comparison(S, Y, title="Signal Comparison", figsize=(12, 10)):
    """
    S: Ground truth sources (n_sources, n_samples)
    Y: Estimated sources (n_sources, n_samples) - Must be pre-aligned/signed
    """
    n = S.shape[0]
    
    # # Normalize to Z-score (Mean=0, Std=1) for visual comparison
    # # This is essential because BSS has scale ambiguity
    # S_norm = (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-9)
    # Y_norm = (Y - Y.mean(axis=1, keepdims=True)) / (Y.std(axis=1, keepdims=True) + 1e-9)
    
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1: axes = [axes]

    for i in range(n):
        # Plot signals
        axes[i].plot(S[i, :], label="Ground Truth", color="#1f77b4", alpha=0.7, linewidth=2)
        axes[i].plot(Y[i, :], label="Estimation", color="#ff7f0e", linestyle="--", linewidth=1.5)
        
        # Calculate local MSE for this channel
        mse = np.mean((S[i, :] - Y[i, :])**2)
        
        axes[i].set_ylabel(f"Source {i}")
        axes[i].legend(loc="upper right", frameon=True, fontsize='small')
        axes[i].grid(True, linestyle=':', alpha=0.6)
        axes[i].set_title(f"Channel {i} - Normalized MSE: {mse:.5f}", fontsize=10)

    plt.xlabel("Time Samples")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
def subplot_1D_signals(
    X, title="", title_fontsize=20, figsize=(10, 5), linewidth=1, colorcode="#050C12"
):
    """Plot the 1D signals (each row from the given matrix)"""
    n = X.shape[0]  # Number of signals

    fig, ax = plt.subplots(n, 1, figsize=figsize)

    for i in range(n):
        ax[i].plot(X[i, :], linewidth=linewidth, color=colorcode)
        ax[i].grid()

    plt.suptitle(title, fontsize=title_fontsize)
    # plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
    plt.draw()