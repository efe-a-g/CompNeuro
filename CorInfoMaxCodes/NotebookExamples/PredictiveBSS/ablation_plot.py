import matplotlib.pyplot as plt
import numpy as np

channels = ["Channel 0", "Channel 1"]
labels = ["Actual", "Simpler ND Update", "neural_lr constant", "lr_W constant"]

means = np.array([
    [28.4593, 29.1636],
    [20.0179, 26.0096],
    [6.2133, 3.9063],
    [27.7609, 27.6059]
])

stds = np.array([
    [2.0814, 1.4946],
    [14.0483, 10.0961],
    [6.0594, 3.3862],
    [3.5117, 3.7865]
])

# 95% CI with Student-t
N = 30
t_crit = 2.045
ci95 = t_crit * stds / np.sqrt(N)

x = np.arange(len(channels))
width = 0.2

plt.figure(figsize=(10,6))

for i in range(len(labels)):
    bars = plt.bar(x + (i - 1.5)*width, means[i], width,
                   yerr=ci95[i], capsize=5, label=labels[i])
    
    for j, bar in enumerate(bars):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h,
                 f"{means[i,j]:.1f}±{ci95[i,j]:.1f}",
                 ha='center', va='bottom', fontsize=8)

plt.xticks(x, channels)
plt.ylabel("SNR (dB)")
plt.title("Per-channel SNR (mean ± 95% CI, Student-t, N=30)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()