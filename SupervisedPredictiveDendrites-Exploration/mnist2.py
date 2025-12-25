from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Config --------------------
hidden_size = 512        # number of hidden neurons (you can tune)
epochs = 10              # epochs per fold (reduce/increase as needed)
n_splits = 3             # KFold splits (reduced to speed up)
random_state = 42
max_samples = 10000      # set to None to use entire MNIST (~70000). Kept small to run faster.
# ------------------------------------------------

# Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_all = mnist['data'].astype(np.float32)
y_all = mnist['target'].astype(np.int64)

if max_samples is not None:
    X_all = X_all[:max_samples]
    y_all = y_all[:max_samples]

# scale to [0,1]
X_all /= 255.0

# optional: normalise each sample to unit norm (like your original normalise for rows)
def normalise_rows(X):
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    # avoid division by zero
    row_norms[row_norms == 0] = 1.0
    return X / row_norms

X_all = normalise_rows(X_all)

# shuffle
X, y_labels = shuffle(X_all, y_all, random_state=random_state)

# one-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y_labels.reshape(-1, 1))

input_dim = X.shape[1]    # 784 for MNIST
n_classes = y.shape[1]    # 10 for MNIST

# -------------------- Utility functions --------------------
def relu(x):
    return np.maximum(0, x)

def g(x):
    # sign returns -1,0,1 -> keep as in original
    return np.sign(x)

def normalise(W):
    # normalise rows (used for weight matrices in original code)
    row_norms = np.linalg.norm(W, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    return W / row_norms
# -----------------------------------------------------------

# -------------------- Learning / Prediction --------------------
def learn(x0, x3, alpha=1.0e-2, beta=1.0e-2, c=1.0):
    global w1f, w1s, w2b, w2f, w2s, w3b, w3f
    x0 = x0.reshape(input_dim, 1)
    x3 = x3.reshape(n_classes, 1)
    x1 = np.zeros((hidden_size, 1))
    x2 = np.zeros((hidden_size, 1))

    MAXITER = 50
    REQDIF = 1e-6
    iteration = 0
    oldx1 = np.zeros_like(x1)
    oldx2 = np.zeros_like(x2)
    dif = 100.0

    v1f = w1f @ x0
    v3b = w3b @ x3

    while dif > REQDIF and iteration < MAXITER:
        iteration += 1
        x1 = relu((v1f+w2b@x2)*0.5 - 0.5 * ((1 - c) / c) * (w1s @ x1))
        x2 = relu((v3b+w2f@x1)*0.5 - 0.5 * ((1 - c) / c) * (w2s @ x2))
        dif = max(np.sum((x1 - oldx1) ** 2),np.sum((x2 - oldx2) ** 2))
        oldx1 = x1.copy()
        oldx2 = x2.copy()

    e1f = x1 - (w1f @ x0)
    w1f = w1f + alpha * (1 - g(x1) / 2) * (e1f @ x0.T)

    e1b = x1 - (w2b @ x2)
    w2b = w2b + alpha * (1 - g(x1) / 2) * (e1b @ x2.T)

    e2f = x2 - (w2f @ x1)
    w2f = w2f + alpha * (1 - g(x2) / 2) * (e2f @ x1.T)

    e3b = x2 - (w3b @ x3)
    w3b = w3b + alpha * (1 - g(x2) / 2) * (e3b @ x3.T)

    e3f = x3 - (w3f @ x2)
    w3f = w3f + alpha * (1 - g(x3) / 2) * (e3f @ x2.T)

    # normalise rows of weight matrices
    w1f = normalise(w1f)
    w2b = normalise(w2b)
    w2f = normalise(w2f)
    w3b = normalise(w3b)
    w3f = normalise(w3f)

    # update somatic lateral weights
    w1s = (1 - beta) * w1s + beta * (x1 @ x1.T)
    np.fill_diagonal(w1s, 0)
    w2s = (1 - beta) * w2s + beta * (x2 @ x2.T)
    np.fill_diagonal(w2s, 0)

    if iteration == MAXITER:
        print("WARNING: relaxation reached MAXITER in learn()")

def predict(x0, c=1.0):
    global w1f, w1s, w2b, w2f, w2s, w3b, w3f
    x0 = x0.reshape(input_dim, 1)
    x1 = np.zeros((hidden_size, 1))
    x2 = np.zeros((hidden_size, 1))
    x3 = np.zeros((n_classes, 1))
    

    MAXITER = 50
    REQDIF = 1e-6
    iteration = 0
    oldx3 = np.zeros_like(x3)
    dif = 100.0

    v1f = w1f @ x0
    v3b = w3b @ x3

    while dif > REQDIF and iteration < MAXITER:
        iteration += 1
        x1 = relu((v1f+w2b@x2)*0.5 - 0.5 * ((1 - c) / c) * (w1s @ x1))
        x2 = relu((v3b+w2f@x1)*0.5 - 0.5 * ((1 - c) / c) * (w2s @ x2))
        x2 = relu(w3f @ x2)
        dif = np.sum((x3 - oldx3) ** 2)
        oldx3 = x3.copy()

    if iteration == MAXITER:
        print("WARNING: relaxation reached MAXITER in predict()")

    return x3
# -----------------------------------------------------------

# -------------------- Weight initialisation --------------------
def init_weights():
    global w1f, w2b, w1s, w2f, w2s, w3b, w3f
    w1f = np.random.randn(hidden_size, input_dim)
    w1f = normalise(w1f)
    w2b = np.random.randn(hidden_size, hidden_size)
    w2b = normalise(w2b)
    w1s = np.zeros((hidden_size, hidden_size))
    w2f = np.random.randn(hidden_size, hidden_size)
    w2f = normalise(w2f)
    w2s = np.zeros((hidden_size, hidden_size))
    w3b = np.random.randn(hidden_size, n_classes)
    w3b = normalise(w3b)
    w3f = np.random.randn(n_classes, hidden_size)
    w3f = normalise(w3f)        


init_weights()
# -----------------------------------------------------------

kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
fold_accuracies = []

for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Starting fold {fold_idx + 1}/{n_splits}")
    # Reinitialize weights for each fold
    init_weights()

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test_labels = y[train_index], y_labels[test_index]

    epoch_acc = []
    for e in range(epochs):
        # training pass (one epoch)
        for i in range(len(X_train)):
            learn(X_train[i], y_train[i])

        # evaluate on test set
        predictions = []
        for j in range(len(X_test)):
            y_hat = predict(X_test[j])
            pred = np.argmax(y_hat)
            predictions.append(pred)

        acc = np.mean(np.array(predictions) == y_test_labels)
        epoch_acc.append(acc)
        print(f"  Fold {fold_idx + 1}, Epoch {e+1}/{epochs}, Accuracy: {acc:.4f}")

    fold_accuracies.append(epoch_acc)

# Average accuracy over folds for each epoch
accuracy = np.mean(fold_accuracies, axis=0)
print("Per-epoch average accuracy across folds:\n", accuracy)

plt.plot(accuracy)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST - relaxation network training accuracy (avg over folds)")
plt.grid(True)
plt.show()
