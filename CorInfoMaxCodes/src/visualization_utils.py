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