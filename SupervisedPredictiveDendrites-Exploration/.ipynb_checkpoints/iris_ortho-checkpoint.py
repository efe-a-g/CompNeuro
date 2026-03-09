from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

iris = load_iris()
X, y_labels = shuffle(iris.data, iris.target, random_state=42)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y_labels.reshape(-1, 1))

def relu(x):
    return np.maximum(0, x)

def g(x):
    return np.sign(x)

def normalise(X):
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / row_norms

def learn(x0, x2, alpha=0.001, gamma =0.001):
    global w1f, w2f
    # Relaxation of the network
    # Inputs:
    #  x0 - fixed values of input layer
    #  w1f - forward weights
    #  w1s - somatic lateral weights
    # Output:
    #  x - activity of neurons in next layer
    x0 = x0.reshape(4, 1)
    x2 = x2.reshape(3, 1)
    N = len(x0)
    x1 = np.zeros((40,1))

    MAXITER = 50
    REQDIF = 0.000001
    iteration = 0
    oldx1 = np.zeros((40,1))
    dif = 100
    
    v1f = w1f @ x0
    v2b = w2f.T @ x2

    while dif > REQDIF and iteration < MAXITER:
        iteration = iteration + 1
        x1 += -gamma * (x1-v1f + x1-v2b)
        dif = np.sum((x1-oldx1)**2)
        oldx1 = x1

    x1 = relu(x1)
    
    e1f = x1 - w1f @ x0
    w1f = w1f + alpha * (1-g(x1)/2) *  e1f @ x0.transpose()

    e2f = x2 - w2f @ x1
    w2f = w2f + alpha * (1-g(x2)/2) * e2f @ x1.transpose()

    w1f = normalise (w1f)
    w2f = normalise(w2f)

    return x1

def predict(x0, gamma=0.001):
    global w1f, w2f
    x0 = x0.reshape(4, 1)
    x2 = np.zeros((3,1))
    x1 = np.zeros((40,1))
    
    MAXITER = 50
    REQDIF = 0.000001
    iteration = 0
    oldx2 = np.zeros((3,1))
    dif = 100
    v1f = w1f @ x0

    while dif > REQDIF and iteration < MAXITER:
        iteration = iteration + 1
        v2b = w2f.T @ x2
        x1 += -gamma * (x1-v1f + x1-v2b)
        x2 = relu(w2f @ x1) 
        dif = np.sum((x2-oldx2)**2)
        oldx2 = x2

    if iteration==MAXITER:
        print("ERROR!")
    return x2

w1f = np.linalg.qr(np.random.randn(40, 4))[0]
w1f = normalise (w1f)

w2f = np.linalg.qr(np.random.randn(3, 40))[0]
w2f = normalise (w2f)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
epochs = 100
accuracy = []

fold_accuracies = []
fold_angles = []

for train_index, test_index in kf.split(X):
    # Reinitialize weights for each fold
    w1f = np.random.randn(40, 4)
    w1f = normalise(w1f)
    w2b = np.random.randn(40, 3)
    w2b = normalise(w2b)
    w1s = np.zeros((40, 40))
    w2f = np.random.randn(3, 40)
    w2f = normalise(w2f)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y_labels[test_index]

    epoch_acc = []
    epoch_angle = []
    for e in range(epochs):
        for i in range(len(X_train)):
            learn(X_train[i], y_train[i])
        predictions = []
        for j in range(len(X_test)):
            y_hat = predict(X_test[j])
            pred = np.argmax(y_hat)
            predictions.append(pred)
        acc = np.mean(np.array(predictions) == y_test)
        epoch_acc.append(acc)
        # Calculate average angle between rows of w2f
        row_angles = []
        for m in range(w2f.shape[0]):
            for n in range(m + 1, w2f.shape[0]):
                u = w2f[m]
                v = w2f[n]
                cos_theta = np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)
                angle = np.arccos(cos_theta) * 180 / np.pi
                row_angles.append(angle)
        avg_angle = np.mean(row_angles)
        epoch_angle.append(avg_angle)
    fold_accuracies.append(epoch_acc)
    fold_angles.append(epoch_angle)

# Average accuracy over folds for each epoch
accuracy = np.mean(fold_accuracies, axis=0)

print(accuracy)
plt.plot(accuracy)
plt.show()
plt.plot(np.mean(fold_angles, axis=0))
plt.xlabel("Epoch")
plt.ylabel("Average angle between rows of w2f (degrees)")
plt.title("Iris - relaxation network training")
plt.grid(True)
plt.show()