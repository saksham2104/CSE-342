import numpy as np
import matplotlib.pyplot as plt

#Dataset 

x = np.random.uniform(0, 1, 100)
epsilon = np.random.normal(0, 0.01, 100)

D = []

for i in range (100):
    D.append((x[i],np.sin(2*np.pi*x[i]) + np.cos(2*np.pi*x[i]) + epsilon[i]))

D_train = D[:80]
D_test = D[80:]

x_train,y_train,x_test,y_test = [],[],[],[]

for i in range(100):
    if i < 80:
        x_train.append(D[i][0])
        y_train.append(D[i][1])
    else:
        x_test.append(D[i][0])
        y_test.append(D[i][1])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


#Gradient Boosting

class DecisionStump:
    def __init__(self):
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, x, residuals, flag):
        thresholds = np.linspace(0, 1, 20)
        min_loss = float('inf')

        for thresh in thresholds:
            left_mask = x <= thresh
            right_mask = x > thresh

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_pred = np.mean(residuals[left_mask])
            right_pred = np.mean(residuals[right_mask])

            predictions = np.where(x <= thresh, left_pred, right_pred)

            if flag == 0:
                curr_loss = np.mean((residuals - predictions) ** 2)
            elif flag == 1:
                curr_loss = np.mean(np.abs(residuals - predictions))
            
            if curr_loss < min_loss:
                min_loss = curr_loss
                self.threshold = thresh
                self.left_value = left_pred
                self.right_value = right_pred

    def predict(self, x):
        return np.where(x <= self.threshold, self.left_value, self.right_value)

rho = 0.01
F_train = np.full(len(x_train), np.mean(y_train))
F_test = np.full(len(x_test), np.mean(y_train))

F_train_1 = np.full(len(x_train), np.mean(y_train))
F_test_1 = np.full(len(x_test), np.mean(y_train))

train_predictions = [F_train.copy()]
test_predictions = [F_test.copy()]
train_losses = []

train_predictions_1 = [F_train_1.copy()]
test_predictions_1 = [F_test_1.copy()]
train_losses_1 = []

for t in range(500):
    # Compute residuals
    residuals = y_train - F_train
    residuals_1 = np.sign(y_train - F_train_1)

    # train decision stumps to residuals
    stump = DecisionStump()
    stump.fit(x_train, residuals ,0)

    stump_1 = DecisionStump()
    stump_1.fit(x_train, residuals_1,1)

    # Gradient descent step
    update = stump.predict(x_train)
    update_1 = stump_1.predict(x_train)
    F_train += rho * update
    F_train_1 += rho * update_1

    # Predict on test set
    update_test = stump.predict(x_test)
    update_test_1 = stump_1.predict(x_test)
    F_test += rho * update_test
    F_test_1 += rho * update_test_1

    # Store predictions and losses
    train_predictions.append(F_train.copy())
    test_predictions.append(F_test.copy())
    train_predictions_1.append(F_train_1.copy())
    test_predictions_1.append(F_test_1.copy())

    loss = np.mean((y_train - F_train) ** 2)
    loss_1 = np.mean(np.abs(y_train - F_train_1))
    train_losses.append(loss)
    train_losses_1.append(loss_1)

rmse_sq = np.sqrt(np.mean((y_test - F_test)  ** 2))
mae_abs = np.mean(np.abs(y_test - F_test_1))

print(f"\nFinal test metrics after {len(train_losses)} boosting rounds:")
print(f"   Mean Squared Error : {rmse_sq:.4f}")
print(f"  Mean Absolute Error: {mae_abs:.4f}")



#  Plot Predictions vs Ground Truth
iterations_to_plot = [0,100,200,300,400,499]  # selected iterations for visualization

plt.figure(figsize=(18, 10))

# Squared Loss - Train Predictions
for i, it in enumerate(iterations_to_plot):
    plt.subplot(2, 3, i + 1)
    plt.scatter(x_train, y_train, color='black', label='Ground Truth', alpha=0.6)
    plt.scatter(x_train, train_predictions[it], color='red', label=f'Predicted (Iter {it})', alpha=0.6)
    plt.title(f'Train (Squared Loss) - Iter {it}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 10))

# Squared Loss - Test Predictions
for i, it in enumerate(iterations_to_plot):
    plt.subplot(2, 3, i + 1)
    plt.scatter(x_test, y_test, color='black', label='Ground Truth', alpha=0.6)
    plt.scatter(x_test, test_predictions[it], color='blue', label=f'Predicted (Iter {it})', alpha=0.6)
    plt.title(f'Test (Squared Loss) - Iter {it}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()

# Absolute Loss - Train Predictions
plt.figure(figsize=(18, 10))
for i, it in enumerate(iterations_to_plot):
    plt.subplot(2, 3, i + 1)
    plt.scatter(x_train, y_train, color='black', label='Ground Truth', alpha=0.6)
    plt.scatter(x_train, train_predictions_1[it], color='green', label=f'Predicted (Iter {it})', alpha=0.6)
    plt.title(f'Train (Absolute Loss) - Iter {it}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()

# Absolute Loss - Test Predictions
plt.figure(figsize=(18, 10))
for i, it in enumerate(iterations_to_plot):
    plt.subplot(2, 3, i + 1)
    plt.scatter(x_test, y_test, color='black', label='Ground Truth', alpha=0.6)
    plt.scatter(x_test, test_predictions_1[it], color='purple', label=f'Predicted (Iter {it})', alpha=0.6)
    plt.title(f'Test (Absolute Loss) - Iter {it}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()
# 2. Plot Training Losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Squared Loss', color='red')
plt.plot(train_losses_1, label='Absolute Loss', color='green')
plt.xlabel('Iterations')
plt.ylabel('Training Loss')
plt.title('Training Loss over Iterations')
plt.legend()
plt.grid(True)
plt.show()
