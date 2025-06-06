#Q-5 -> 5 fold cross validation
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


n = 100
x = np.random.uniform(0, 2*np.pi, n)
y_true = np.sin(x)
noise = np.random.normal(0, 0.1, n)
y = y_true + noise
 
def poly_features(x, d):
    x_poly = np.ones((len(x), d + 1))
    for i in range(len(x)):  
        for j in range(1, d + 1):  
            x_poly[i][j] = x_poly[i][j - 1]*x[i]  

    return x_poly


def lin_reg(x, y):
    matrice_A=x.T @ x
    det_a=np.linalg.det(matrice_A)
    if(det_a == 0):
        return np.linalg.pinv(x.T @ x)@ x.T @ y 
    return np.linalg.inv(x.T @ x)@ x.T @ y 

def predict(x, w):
    return x @ w

def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

#  Split data
def k_fold(n):
    k=5
    idx = np.arange(n)
    np.random.shuffle(idx)
    
    fold_sz = [20]*5
    
    folds = []
    start = 0
    for size in fold_sz:
        test_idx = idx[start:start + size]
        train_idx = np.concatenate((idx[:start], idx[start + size:]))
        folds.append((train_idx, test_idx))
        start += size
    
    return folds

degrees = [1, 2, 3, 4]
k = 5
cv_scores = {}
train_scores = {}


for d in degrees: 
    test_mses = []
    train_mses = []
    
    for train_idx, test_idx in k_fold(n):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        x_train_poly = poly_features(x_train, d)
        x_test_poly = poly_features(x_test, d)
        
        w = lin_reg(x_train_poly, y_train)
        
        y_train_pred = predict(x_train_poly, w)
        y_test_pred = predict(x_test_poly, w)
        
        train_mses.append(mse(y_train, y_train_pred))
        test_mses.append(mse(y_test, y_test_pred))
    
    cv_scores[d] = np.mean(test_mses)
    train_scores[d] = np.mean(train_mses)

best_d = 1
mini = 1000000 

for d in cv_scores:
    if cv_scores[d] < mini:
        mini = cv_scores[d]
        best_d = d


print("Train & Test MSE for each degree:")
for d in degrees:
    print(f"Degree {d}: Train MSE = {train_scores[d]:.6f}, Test MSE = {cv_scores[d]:.6f}")

print(f"\nBest degree: {best_d}")


x_poly = poly_features(x, best_d)
w_best = lin_reg(x_poly, y)


x_plot = np.linspace(0, 2*np.pi, 1000)
y_plot = np.sin(x_plot)
x_plot_poly = poly_features(x_plot, best_d)
y_pred_plot = predict(x_plot_poly, w_best)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', alpha=0.5, label='Noisy Data')
plt.plot(x_plot, y_plot, color='green', label='True Function')
plt.plot(x_plot, y_pred_plot, color='red', linestyle='--', label=f'Poly (degree {best_d})')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Polynomial Regression (Best: deg {best_d})')
plt.legend()
plt.grid()
plt.show()
