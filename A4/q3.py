import numpy as np

np.random.seed(42)

# Dataset 
I = np.eye(2)

label_0 = np.random.multivariate_normal(mean=[-1, -1], cov=I, size=10)
label_1 = np.random.multivariate_normal(mean=[1, 1], cov=I, size=10)

y0 = np.zeros((10, 1))
y1 = np.ones((10, 1))


X = np.vstack((label_0, label_1))
y = np.vstack((y0, y1))

X_train = np.vstack((label_0[:5], label_1[:5]))
y_train = np.vstack((y0[:5], y1[:5]))

X_test = np.vstack((label_0[5:], label_1[5:]))
y_test = np.vstack((y0[5:], y1[5:]))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_dash(x):
    s = sigmoid(x)
    return s * (1 - s)

# Randomize iniial parametres ->  weights and biases
w1 = np.random.randn(2, 1)
b1 = np.random.randn(1)

w2 = np.random.randn(1, 1)
b2 = np.random.randn(1)


lr = 0.1 # Learning rate

for iteration in range(1000):
    total_loss = 0
    for x, y_true in zip(X_train, y_train):
        x = x.reshape(1,2)       
        y_true = y_true.reshape(1, 1) 
        # Forward pass
        z1 = x @ w1 + b1           # (1, 1)
        a1 = sigmoid(z1)           # (1, 1)
        y_pred = a1 @ w2 + b2      # (1, 1)

        #  Loss (MSE)
        loss = 0.5 * (y_pred - y_true) ** 2
        total_loss += loss.item()

        #  Backprop and update using gradient descent
        dL_dy = y_pred - y_true         # (1, 1)
        dL_dw2 = a1.T @ dL_dy           # (1, 1)
        dL_db2 = dL_dy.item()

        dL_da1 = dL_dy @ w2.T           # (1, 1)
        da1_dz1 = sigmoid_dash(z1)      # (1, 1)
        dL_dz1 = dL_da1 * da1_dz1       # (1, 1)
        dL_dw1 = x.T @ dL_dz1           # (2, 1)
        dL_db1 = dL_dz1.item()

        #update -> W = W - lr*dl/dw
        w2 -= lr * dL_dw2
        b2 -= lr * dL_db2
        w1 -= lr * dL_dw1
        b1 -= lr * dL_db1

    if iteration % 300 == 0:
        avg_loss = total_loss / len(X_train)
        print(f"iteration {iteration}: Training Loss = {avg_loss:.4f}")
        print(f"  w1 = {w1.ravel()}, b1 = {b1.item():.4f}")
        print(f"  w2 = {w2.ravel()}, b2 = {b2.item():.4f}\n")

    if iteration == 999:
        avg_loss = total_loss / len(X_train)
        print(f"iteration {iteration}: Training Loss = {avg_loss:.4f}")

def predict(X):
    z1 = X @ w1 + b1
    a1 = sigmoid(z1)
    y_hat = a1 @ w2 + b2
    return y_hat

y_pred = predict(X_test)
mse = np.mean((y_pred - y_test) ** 2)
print(f"\nTest MSE: {mse:.4f}")
