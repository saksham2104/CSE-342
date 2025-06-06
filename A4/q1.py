import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from numpy.linalg import svd

def pca_fit(X, k):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    _, _, Vt = svd(Xc, full_matrices=False)
    V_k = Vt[:k]     
    return mu, V_k

def pca_transform(X, mu, V_k):
    return (X - mu) @ V_k.T

#Load dataset and apply PCA reduce to 5D
def load_mnist_01(train_per_class=1000, val_frac=0.1):
    (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

    mask_tr = np.isin(y_train_full, (0, 1))
    mask_te = np.isin(y_test_full,  (0, 1))
    X_tr_01, y_tr_01 = X_train_full[mask_tr], y_train_full[mask_tr]
    X_te_01, y_te_01 = X_test_full[mask_te],  y_test_full[mask_te]

    idx0 = np.where(y_tr_01 == 0)[0][:train_per_class]   
    idx1 = np.where(y_tr_01 == 1)[0][:train_per_class]   
    idx  = np.concatenate([idx0, idx1])                  

    X_small, y_small = X_tr_01[idx], y_tr_01[idx]
    
    n_train = int(len(X_small) * (1 - val_frac)) 
    X_train, X_val = X_small[:n_train], X_small[n_train:]
    y_train, y_val = y_small[:n_train], y_small[n_train:]


    X_tr_f = X_train.reshape(len(X_train), -1).astype(np.float32) / 255.0
    X_val_f = X_val.reshape(len(X_val), -1).astype(np.float32) / 255.0
    X_te_f = X_te_01.reshape(len(X_te_01), -1).astype(np.float32) / 255.0

    #PCA
    mu, V_k = pca_fit(X_tr_f, k=5)      # fit on TRAIN only
    X_tr_p  = pca_transform(X_tr_f,  mu, V_k)
    X_val_p = pca_transform(X_val_f, mu, V_k)
    X_te_p  = pca_transform(X_te_f,  mu, V_k)


    y_tr_b = np.where(y_train == 0, -1, +1)
    y_val_b = np.where(y_val == 0, -1, +1)
    y_te_b = np.where(y_te_01 == 0, -1, +1)

    return (X_tr_p, y_tr_b,
            X_val_p, y_val_b,
            X_te_p,  y_te_b)


def best_stump(X, y, w): #find best stump 

    n, d = X.shape
    best = (None, None, None, 1.0)   

    for dim in range(d):
        x = X[:, dim]
        mn, mx = x.min(), x.max()
        steps = np.linspace(mn, mx, 4)[1:-1]   

        for thr in steps:
            for polarity in (+1, -1):
                preds = polarity * np.where(x < thr, +1, -1)
                miss  = preds != y
                eps   = w[miss].sum()
                if eps < best[3]: #3 cuts 
                    best = (dim, thr, polarity, eps)

    return best 

def adaboost(X, y, X_val, y_val, X_test, y_test,rounds=200):
                   
    n = len(X)
    w = np.full(n, 1 / n)
    learners = []        

    loss_tr, loss_val, loss_te = [], [], []
    err_train = []

    def exp_loss(X_, y_, learners_):
        F = np.zeros(len(X_))
        for dim, thr, pol, b in learners_:
            F += b * pol * np.where(X_[:, dim] < thr, +1, -1)
        return np.exp(-y_ * F).mean()   

    def zero_one_loss(X_, y_, learners_):
        F = np.zeros(len(X_))
        for dim, thr, pol, b in learners_:
            F += b * pol * np.where(X_[:, dim] < thr, +1, -1)
        return (np.sign(F) != y_).mean() 

    for m in range(rounds):
        dim, thr, pol, eps = best_stump(X, y, w)

        eps = np.clip(eps, 1e-12, 1 - 1e-12)
        beta = 0.5 * np.log((1 - eps) / eps)

        learners.append((dim, thr, pol, beta))

        preds = pol * np.where(X[:, dim] < thr, +1, -1)
        w *= np.exp(-beta * y * preds)
        w /= w.sum()

        loss_tr.append(exp_loss(X,y,learners))
        loss_val.append(exp_loss(X_val, y_val,learners))
        loss_te.append(exp_loss(X_test, y_test,learners))
        err_train.append(zero_one_loss(X, y,learners))

        if m % 40 == 0 : 
            print(f"Round {m+1:3d}: eps={eps:.3f}, beta={beta:.3f}, "
                f"train err={err_train[-1]*100:.4f}%")

    return learners, loss_tr, loss_val, loss_te, err_train

def plot(loss_tr, loss_val, loss_te, err_tr):
    rounds = np.arange(1, len(loss_tr) + 1)
    plt.figure(figsize=(10, 4))

    # Plot-1 Loss
    plt.subplot(1, 2, 1)
    plt.plot(rounds, loss_tr, label='train')
    plt.plot(rounds, loss_val, label='val')
    plt.plot(rounds, loss_te, label='test')
    plt.xlabel("Boosting rounds")
    plt.ylabel("Exponential loss")
    plt.title("Loss vs rounds")
    plt.legend()

    # Plot-2 Training error
    plt.subplot(1, 2, 2)
    plt.plot(rounds, err_tr, color='tab:red')
    plt.xlabel("Boosting rounds")
    plt.ylabel("Training error (0‑1)")
    plt.title("Training error vs rounds")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_mnist_01()

    learners, loss_tr, loss_val, loss_te, err_tr = adaboost(X_tr, y_tr, X_val, y_val, X_te, y_te, rounds=200)

    def predict(X):
        F = np.zeros(len(X))
        for dim, thr, pol, b in learners:
            F += b * pol * np.where(X[:, dim] < thr, +1, -1) #final function which is sum of all weak learners
        return np.sign(F)

    test_acc = (predict(X_te) == y_te).mean() * 100 
    print(f"\nFinal test accuracy: {test_acc:.2f}%") #final test accuracy

    plot(loss_tr, loss_val, loss_te, err_tr)
