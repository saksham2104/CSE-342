import tensorflow as tf
import numpy as np
import heapq  # Priority queue (min-heap)

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Filter classes
    classes = [0, 1, 2]
    train_arr = np.isin(y_train, classes)
    test_arr = np.isin(y_test, classes)

    x_train, y_train = x_train[train_arr], y_train[train_arr]  
    x_test, y_test = x_test[test_arr], y_test[test_arr]  

    # Normalize images
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0  
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    return x_train, y_train, x_test, y_test  

def covariance_matrix(X):
    N = X.shape[0]
    mu = np.mean(X, axis=0)
    X_c = X - mu
    cov_matrix = (X_c.T @ X_c)/(N - 1)
    return cov_matrix

def pca(X, variance_threshold=0.95):
    num_samples, num_features = X.shape
    mu = np.mean(X, axis=0)  
    X_c = X - mu  

    cov_matrix = covariance_matrix(X_c)  

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    eigen_heap = [(-eigenvalues[i], i) for i in range(len(eigenvalues))] 
    heapq.heapify(eigen_heap)

    sorted_indices = [heapq.heappop(eigen_heap)[1] for _ in range(len(eigenvalues))]  
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    if variance_threshold is None:  
        num_components = num_features 
    else:
        total_variance = np.sum(sorted_eigenvalues)  
        variance_explained = np.cumsum(sorted_eigenvalues) / total_variance  

        num_components = np.argmax(variance_explained >= variance_threshold) + 1  

    top_eigenvectors = sorted_eigenvectors[:, :num_components]  

    X_pca = np.dot(X_c, top_eigenvectors)

    return X_pca, top_eigenvectors, mu 

def project_test_data(x_test, top_eigenvectors, mu):
    """Projects new test data into the PCA-reduced space."""
    Xc_test = x_test - mu  
    Y_test = np.dot(Xc_test, top_eigenvectors)  
    return Y_test


### FDA using Priority Queue
def compute_scatter_matrices(x_train, y_train):
    classes = [0, 1, 2]  
    mu = np.mean(x_train, axis=0).reshape(-1, 1)  

    SB = np.zeros((x_train.shape[1], x_train.shape[1])) 
    SW = np.zeros((x_train.shape[1], x_train.shape[1]))

    for i in classes:
        curr = x_train[y_train == i]  
        curr_count = curr.shape[0]  
        curr_mean = np.mean(curr, axis=0).reshape(-1, 1)  

       
        mean_diff = curr_mean - mu
        SB += curr_count * (mean_diff @ mean_diff.T)

        class_scatter = covariance_matrix(curr)  
        SW += class_scatter
    
    return SB, SW


def compute_fda(SB, SW):
    I = np.eye(SW.shape[0])  # Identity matrix
    SW_inv = np.linalg.pinv(SW + 0.001 * I) 

    SW_inv_SB = SW_inv @ SB  # Compute SW^-1 * SB
    print("Shape of SW^-1 * SB:", SW_inv_SB.shape)  # Corrected shape printing

    eigenvalues, eigenvectors = np.linalg.eig(SW_inv_SB)

    eigen_heap = [(-eigenvalues[i], i) for i in range(len(eigenvalues))]  # Max-heap (negate values)
    heapq.heapify(eigen_heap)

    sorted_indices = [heapq.heappop(eigen_heap)[1] for _ in range(2)]  
    W = eigenvectors[:, sorted_indices]

    return W

def project_fda(X, W):
    return np.dot(X, W)

# Load dataset
x_train, y_train, x_test, y_test = load_data()

# Apply PCA
Y_train_pca, top_eigenvectors, mu = pca(x_train)  # Correct function call

# Project test data using PCA
Y_test_pca = project_test_data(x_test, top_eigenvectors, mu)

# Compute scatter matrices explicitly using PCA-transformed data
SB, SW = compute_scatter_matrices(Y_train_pca, y_train)

# Compute FDA projection matrix W
W_fda = compute_fda(SB, SW)

# Project PCA-transformed data using FDA
Y_train_fda = project_fda(Y_train_pca, W_fda)
Y_test_fda = project_fda(Y_test_pca, W_fda)


# Apply FDA on training and test sets
SB, SW = compute_scatter_matrices(x_train, y_train)
W_fda = compute_fda(SB, SW)
X_train_fda = project_fda(x_train, W_fda)
X_test_fda = project_fda(x_test, W_fda)


print(f"Original Dimensionality: {x_train.shape[1]}") 
print(f"PCA Reduced Dimensionality: {Y_train_pca.shape[1]}")  
print(f"FDA Reduced Dimensionality: {Y_train_fda.shape[1]}")

print(f"Transformed Train Shape (FDA): {Y_train_fda.shape}")
print(f"Transformed Test Shape (FDA): {Y_test_fda.shape}")

import numpy as np

# Compute class priors
def compute_class_priors(y_train):
    classes = np.unique(y_train)
    priors = {c: np.mean(y_train == c) for c in classes}
    return priors

def train_lda(x_train, y_train):
    classes = np.unique(y_train)
    means = {c: np.mean(x_train[y_train == c], axis=0) for c in classes}
    
    cov_matrix = np.cov(x_train.T)  
    cov_inv = np.linalg.pinv(cov_matrix) 
    
    priors = compute_class_priors(y_train)
    
    return means, cov_inv, priors

def predict_lda(x, means, cov_inv, priors):
    predictions = []
    for sample in x:
        scores = {}
        for c in means:
            mean = means[c]
            likelihood = -0.5 * np.dot((sample - mean).T, np.dot(cov_inv, (sample - mean)))
            prior_term = np.log(priors[c])
            scores[c] = likelihood + prior_term
        predictions.append(max(scores, key=scores.get))
    return np.array(predictions)

def train_qda(x_train, y_train):
    classes = np.unique(y_train)
    means = {c: np.mean(x_train[y_train == c], axis=0) for c in classes}
    cov_matrices = {c: np.cov(x_train[y_train == c].T) for c in classes}
    
    priors = compute_class_priors(y_train)
    
    return means, cov_matrices, priors

def predict_qda(x, means, cov_matrices, priors):
    predictions = []
    for sample in x:
        scores = {}
        for c in means:
            mean = means[c]
            cov = cov_matrices[c]
            cov_inv = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-4)  # Regularization
            log_det = np.log(np.linalg.det(cov) + 1e-6)  # Log determinant
            likelihood = -0.5 * (np.dot((sample - mean).T, np.dot(cov_inv, (sample - mean))) + log_det)
            prior_term = np.log(priors[c])
            scores[c] = likelihood + prior_term
        predictions.append(max(scores, key=scores.get))
    return np.array(predictions)

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

# Training and evaluation pipeline
def evaluate_model(X_train, X_test, y_train, y_test):
    # Train LDA
    means_lda, cov_inv_lda, priors_lda = train_lda(X_train, y_train)
    y_train_pred_lda = predict_lda(X_train, means_lda, cov_inv_lda, priors_lda)
    y_test_pred_lda = predict_lda(X_test, means_lda, cov_inv_lda, priors_lda)
    
    # Train QDA
    means_qda, cov_matrices_qda, priors_qda = train_qda(X_train, y_train)
    y_train_pred_qda = predict_qda(X_train, means_qda, cov_matrices_qda, priors_qda)
    y_test_pred_qda = predict_qda(X_test, means_qda, cov_matrices_qda, priors_qda)
    
    # Compute Accuracy
    lda_train_acc = compute_accuracy(y_train, y_train_pred_lda)
    lda_test_acc = compute_accuracy(y_test, y_test_pred_lda)
    qda_train_acc = compute_accuracy(y_train, y_train_pred_qda)
    qda_test_acc = compute_accuracy(y_test, y_test_pred_qda)
    
    print(f"LDA Train Accuracy: {lda_train_acc:.2f}%")
    print(f"LDA Test Accuracy: {lda_test_acc:.2f}%")
    print(f"QDA Train Accuracy: {qda_train_acc:.2f}%")
    print(f"QDA Test Accuracy: {qda_test_acc:.2f}%")

# PCA + LDA
X_pca_90, top_eigenvectors_90, mu_90 = pca(x_train, variance_threshold=0.90)
X_test_pca_90 = project_test_data(x_test, top_eigenvectors_90, np.mean(x_train, axis=0))
SB_90, SW_90 = compute_scatter_matrices(X_pca_90, y_train)
W_fda_90 = compute_fda(SB_90, SW_90)
X_fda_90 = project_fda(X_pca_90, W_fda_90)
X_test_fda_90 = project_fda(X_test_pca_90, W_fda_90)

means_lda_90, cov_inv_lda_90, priors_lda_90 = train_lda(X_fda_90, y_train)
y_train_pred_lda_90 = predict_lda(X_fda_90, means_lda_90, cov_inv_lda_90, priors_lda_90)
y_test_pred_lda_90 = predict_lda(X_test_fda_90, means_lda_90, cov_inv_lda_90, priors_lda_90)
lda_train_acc_90 = compute_accuracy(y_train, y_train_pred_lda_90)
lda_test_acc_90 = compute_accuracy(y_test, y_test_pred_lda_90)

print(f"LDA Train Accuracy (90% Variance): {lda_train_acc_90:.2f}%")
print(f"LDA Test Accuracy (90% Variance): {lda_test_acc_90:.2f}%")

# PCA (First 2 Components) + LDA
X_pca_2, top_eigenvectors_2,mu_pca_2 = pca(x_train, variance_threshold=None)
X_pca_2 = X_pca_2[:, :2]
X_test_pca_2 = project_test_data(x_test, top_eigenvectors_2[:, :2], np.mean(x_train, axis=0))
SB_2, SW_2 = compute_scatter_matrices(X_pca_2, y_train)
W_fda_2 = compute_fda(SB_2, SW_2)
X_fda_2 = project_fda(X_pca_2, W_fda_2)
X_test_fda_2 = project_fda(X_test_pca_2, W_fda_2)

means_lda_2, cov_inv_lda_2, priors_lda_2 = train_lda(X_fda_2, y_train)
y_train_pred_lda_2 = predict_lda(X_fda_2, means_lda_2, cov_inv_lda_2, priors_lda_2)
y_test_pred_lda_2 = predict_lda(X_test_fda_2, means_lda_2, cov_inv_lda_2, priors_lda_2)
lda_train_acc_2 = compute_accuracy(y_train, y_train_pred_lda_2)
lda_test_acc_2 = compute_accuracy(y_test, y_test_pred_lda_2)

print(f"LDA Train Accuracy (First 2 PCA Components): {lda_train_acc_2:.2f}%")
print(f"LDA Test Accuracy (First 2 PCA Components): {lda_test_acc_2:.2f}%")


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(15, 6))

plt.subplot(121)
for i, label in enumerate(['0', '1', '2']):
    mask = y_train == i
    plt.scatter(X_pca_2[mask, 0], X_pca_2[mask, 1], 
                label=f'Digit {label}',
                alpha=0.6)

plt.title('PCA Projection (First 2 Components)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True)

plt.subplot(122)
for i, label in enumerate(['0', '1', '2']):
    mask = y_train == i
    plt.scatter(X_fda_2[mask, 0], X_fda_2[mask, 1], 
                label=f'Digit {label}',
                alpha=0.6)

plt.title('FDA Projection')
plt.xlabel('First Discriminant')
plt.ylabel('Second Discriminant')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 6))

plt.subplot(121)
for i, label in enumerate(['0', '1', '2']):
    mask = y_test == i
    plt.scatter(X_test_pca_2[mask, 0], X_test_pca_2[mask, 1], 
                label=f'Digit {label}',
                alpha=0.6)

plt.title('PCA Projection - Test Data')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True)

plt.subplot(122)
for i, label in enumerate(['0', '1', '2']):
    mask = y_test == i
    plt.scatter(X_test_fda_2[mask, 0], X_test_fda_2[mask, 1], 
                label=f'Digit {label}',
                alpha=0.6)

plt.title('FDA Projection - Test Data')
plt.xlabel('First Discriminant')
plt.ylabel('Second Discriminant')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
