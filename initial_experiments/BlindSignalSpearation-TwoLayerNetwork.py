import numpy as np
from itertools import permutations, product
import matplotlib.pyplot as plt

class DataGenerator:
    def __init__(self, N=2, Npoints=100, mean=np.array([0,0]), cov=np.array([[1,1],[1,1]])):
        self.N = N
        self.Npoints = Npoints
        self.mean = mean
        self.cov = cov
    
    def generate_mixed_sources(self, N, d, m, random_state=None):
        """
        Generate d-dimensional sources,
        then mix them with an m x d Gaussian random matrix.

        Parameters
        ----------
        N : int
            Number of samples per source.
        d : int
            Number of sources (dimensions of input).
        m : int
            Number of mixtures (rows of mixing matrix A).
        random_state : int or None
            Seed for reproducibility.

        Returns
        -------
        U : ndarray, shape (N, d)
        X : ndarray, shape (N, m)
        A : ndarray, shape (m, d)
        """
        rng = np.random.default_rng(random_state)
        
        U = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(N,d))
        # Generate a random orthonormal matrix for A
        A = rng.standard_normal(size=(m, d))
        X = U @ A.T  # shape (N, m)
        """std = X.std(axis=0)

        # Avoid divide-by-zero errors
        std[std == 0] = 1

        # Standardize
        X = X / std"""
        
        return U, X, A
    
    def generate_input(self):
        u0, x0, a0 = self.generate_mixed_sources(N=self.Npoints, d=self.N, m=self.N)
        return u0, x0, a0

class BlindSignalSeparationNetwork:
    def __init__(self, N):
        self.N = N
        self.w1f, self.w1s, self.var = self.initialize_w()
    
    def normalise(self, X):
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / row_norms
    
    def initialize_w(self):
        w1f = np.random.randn(self.N, self.N)
        w1f = self.normalise(w1f)
        w1s = np.zeros((self.N, self.N))
        var = np.zeros(self.N)
        return w1f, w1s, var
    
    def activation(self, x):
        return np.clip(x, -np.sqrt(3), np.sqrt(3))
    
    def clip(self, W):
        for row in W:
            norm = np.linalg.norm(row)
            if norm < 1:
                row /= norm
        return W
    
    def relax(self, x0):
        N = len(x0)
        x0 = x0.reshape(N, 1)
        x = [x0, np.zeros((N,1))]
        v1f = self.w1f @ x[0]
        for i in range(100):
            x[1] = (v1f - self.w1s @ x[1])
        x[1][0] = x[1][0] / (2 * self.var[0] - 1)
        x[1][1] = x[1][1] / (2 * self.var[1] - 1)
        x[1] = self.activation(x[1])
        return x
    
    def update_w(self, x, alpha=1e-1, beta=1e-3):
        e1f = x[1] - self.w1f @ x[0]
        self.w1f = self.w1f + beta * e1f @ x[0].transpose()
        self.w1f = self.clip(self.w1f)
        self.w1s = (1 - alpha) * self.w1s + alpha * x[1] @ x[1].transpose()
        np.fill_diagonal(self.w1s, 0)
        self.var = (1 - alpha) * self.var + alpha * np.array([x[1][0]**2, x[1][1]**2]).flatten()
        return self.w1f, self.w1s, self.var

class Trainer:
    def __init__(self, network, data_gen, epochs=10, ITER=10, Npoints=2000):
        self.network = network
        self.data_gen = data_gen
        self.epochs = epochs
        self.ITER = ITER
        self.Npoints = Npoints
    
    def train_epoch(self, u0, x0):
        cosw = np.zeros(self.ITER)
        inh = np.zeros(self.ITER)
        cor = np.zeros(self.ITER)
        for it in range(self.ITER):
            cors = []
            for i in range(self.Npoints):
                x = self.network.relax(x0[i])
                cors.append(x[1][0] * x[1][1])
            cor[it] = np.mean(cors)
            cosw[it] = np.dot(self.network.w1f[0], self.network.w1f[1])
            inh[it] = self.network.w1s[1, 0]
            for i in range(self.Npoints):
                x = self.network.relax(x0[i])
                self.network.update_w(x)
        return cosw, inh, cor
    
    def evaluate_snr(self, u0, x0, a0):
        best_snr = -np.inf
        best_w = None
        for perm in permutations(range(self.network.N)):
            for signs in product([1, -1], repeat=self.network.N):
                W = self.network.w1f[list(perm), :] * np.array(signs)[:, None]
                approx_u0 = (W @ x0.T).T
                snr_value = 10 * np.log10(np.linalg.norm(u0)**2 / np.linalg.norm(u0 - approx_u0)**2)
                if snr_value > best_snr:
                    best_snr = snr_value
                    best_w = W
        return best_snr, best_w

def plot_results(u0, x0, transformed, snr):
    def scatter2d(P, label_):
        plt.plot(P[:, 0], P[:, 1], 'o', linestyle='None', markersize=4, label=label_)
    scatter2d(u0, "S")
    scatter2d(x0, "X")
    scatter2d(transformed, "approx S")
    plt.legend()
    plt.show()
    plt.plot(snr)
    print(np.mean(snr))
    plt.show()

# Main execution
data_gen = DataGenerator(N=2, Npoints=2000)
network = BlindSignalSeparationNetwork(N=2)
trainer = Trainer(network, data_gen, epochs=10, ITER=10, Npoints=2000)
snr = []

for e in range(trainer.epochs):
    u0, x0, a0 = data_gen.generate_input()
    cosw, inh, cor = trainer.train_epoch(u0, x0)
    best_snr, best_w = trainer.evaluate_snr(u0, x0, a0)
    print(best_w @ a0)
    print(best_snr)
    snr.append(best_snr)
    
    transformed = np.zeros((trainer.Npoints, trainer.network.N))
    for i in range(trainer.Npoints):
        transformed[i] = network.activation(best_w @ x0[i]).flatten()
    plot_results(u0, x0, transformed, snr)

"""
print(w1f)
print(w1s)
print(var)
plt.plot (cosw)
plt.xlabel ('Training iteration')
plt.ylabel ('Cosine of angle between weights')
plt.figure()

plt.plot(inh)
plt.xlabel ('Training iteration')
plt.ylabel ('Inhibitory connection')

plt.figure()

plt.plot (cor)
plt.xlabel ('Training iteration')
plt.ylabel ('Correlation')

plt.show()
"""