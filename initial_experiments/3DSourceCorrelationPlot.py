import numpy as np
import matplotlib.pyplot as plt

class DataGenerator:
    def __init__(self, N=3, Npoints=2000):
        self.N = N
        self.Npoints = Npoints
    
    def generate_mixed_sources(self, N, d, m, random_state=None):
        rng = np.random.default_rng(random_state)
        U = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(N, d))
        A = 0.5 * rng.standard_normal(size=(m, d))
        X = U @ A.T
        return U, X, A
    
    def generate_input(self):
        _, x0, _ = self.generate_mixed_sources(N=self.Npoints, d=self.N, m=self.N)
        return x0

class BlindSignalSeparationNetwork:
    def __init__(self, N):
        self.N = N
        self.w1f, self.w1s = self.initialize_w()
    
    def normalise(self, X):
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / row_norms
    
    def activation(self, x):
        return np.clip(x, -np.sqrt(3), np.sqrt(3))
    
    def initialize_w(self):
        w1f = np.random.randn(self.N, self.N)
        w1f = self.normalise(w1f)
        w1s = np.zeros((self.N, self.N))
        return w1f, w1s
    
    def relax(self, x0):
        N = len(x0)
        x0 = x0.reshape(N, 1)
        x = [x0, np.zeros((N, 1))]
        MAXITER = 50
        REQDIF = 0.000001
        iteration = 0
        oldx1 = np.zeros((N, 1))
        dif = 100
        v1f = self.w1f @ x[0]
        while dif > REQDIF and iteration < MAXITER:
            iteration += 1
            x[1] = v1f - self.w1s @ x[1]
            dif = np.sum((x[1] - oldx1)**2)
            oldx1 = x[1]
        if iteration == MAXITER:
            print("ERROR!")
        x[1] = self.activation(x[1])
        return x
    
    def update_w(self, x, alpha=0.001):
        e1f = x[1] - self.w1f @ x[0]
        self.w1f = self.w1f + alpha * e1f @ x[0].transpose()
        self.w1s = (1 - alpha) * self.w1s + alpha * x[1] @ x[1].transpose()
        np.fill_diagonal(self.w1s, 0)
        return self.w1f, self.w1s

class Trainer:
    def __init__(self, network, data_gen, epochs=100, ITER=100, Npoints=2000):
        self.network = network
        self.data_gen = data_gen
        self.epochs = epochs
        self.ITER = ITER
        self.Npoints = Npoints
    
    def train_epoch(self, x0):
        cor1 = np.zeros(self.ITER)
        cor2 = np.zeros(self.ITER)
        cor3 = np.zeros(self.ITER)
        for it in range(self.ITER):
            cors1 = []
            cors2 = []
            cors3 = []
            for i in range(self.Npoints):
                x = self.network.relax(x0[i])
                cors1.append(x[1][0] * x[1][1])
                cors2.append(x[1][1] * x[1][2])
                cors3.append(x[1][0] * x[1][2])
            cor1[it] = np.mean(cors1)
            cor2[it] = np.mean(cors2)
            cor3[it] = np.mean(cors3)
            for i in range(self.Npoints):
                x = self.network.relax(x0[i])
                self.network.update_w(x)
        return cor1[-1], cor2[-1], cor3[-1]

# Main execution
data_gen = DataGenerator(N=3, Npoints=2000)
network = BlindSignalSeparationNetwork(N=3)
trainer = Trainer(network, data_gen, epochs=30, ITER=25, Npoints=2000)

c1 = []
c2 = []
c3 = []

for e in range(trainer.epochs):
    x0 = data_gen.generate_input()
    final_cor1, final_cor2, final_cor3 = trainer.train_epoch(x0)
    c1.append(final_cor1)
    c2.append(final_cor2)
    c3.append(final_cor3)

print(np.mean(c1), np.mean(c2), np.mean(c3))
plt.plot(c1)
plt.plot(c2)
plt.plot(c3)
plt.xlabel('Epochs')
plt.ylabel('Final correlation')
plt.legend(['correlation 1-2', 'correlation 2-3', 'correlation 1-3'])
plt.show()

