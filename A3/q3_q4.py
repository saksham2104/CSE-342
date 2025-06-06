import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

np.random.seed(20)

class Node:
    def __init__(self, f=None, t=None, l=None, r=None, v=None):
        self.feature = f
        self.threshold = t
        self.left = l
        self.right = r
        self.value = v

class DecisionTree:
    def __init__(self, max_d=4, min_s=1, feat_sub=None):
        self.max_depth = max_d
        self.min_samples = min_s
        self.root = None
        self.feature_subset = feat_sub #for RF

    #gini impurity
    def gini(self, y):
        cls , cnt = np.unique(y, return_counts=True)
        probs = cnt / cnt.sum()
        return 1 - np.sum(probs ** 2)

    def best_split(self, X, y):
        best_g = float('inf')
        best_f, best_t = None, None
        n, m = X.shape

        if self.feature_subset and self.feature_subset < m:
            feats = np.random.choice(m, self.feature_subset, replace=False) #dont repeat same feature twice
        else:
            feats = range(m)

        for f in feats:
            vals = np.unique(X[:, f])
            for t in vals:
                l_mask = X[:, f] == t
                r_mask = ~l_mask

                if sum(l_mask) == 0 or sum(r_mask) == 0:
                    continue

                #training on gini
                g_l = self.gini(y[l_mask])
                g_r = self.gini(y[r_mask])
                g_split = (sum(l_mask) * g_l + sum(r_mask) * g_r) / n

                if g_split < best_g:
                    best_g = g_split
                    best_f = f
                    best_t = t

        return best_f, best_t

    def build(self, X, y, d=0):
        n, m = X.shape
        cls = np.unique(y)

        #stopping conditions
        if d >= self.max_depth or n < self.min_samples or len(cls) == 1:
            leaf = cls[0] if len(cls) == 1 else max(cls, key=list(y).count)
            return Node(v=leaf)

        f, t = self.best_split(X, y)

        if f is None:
            return Node(v=max(cls, key=list(y).count))

        l_mask = X[:, f] == t
        r_mask = ~l_mask

        #recursive method
        l_sub = self.build(X[l_mask], y[l_mask], d + 1)
        r_sub = self.build(X[r_mask], y[r_mask], d + 1)

        return Node(f=f, t=t, l=l_sub, r=r_sub)

    def fit(self, X, y):
        self.root = self.build(X, y)

    def pred_sample(self, node, x):
        if node.value is not None:
            return node.value
        if x[node.feature] == node.threshold:
            return self.pred_sample(node.left, x)
        else:
            return self.pred_sample(node.right, x)
    #prediciton
    def predict(self, X):
        preds = []
        for x in X:
            preds.append(self.pred_sample(self.root, x))
        return np.array(preds)

class Bagging:
    #bagging of 10 trees
    def __init__(self, n_trees=10, max_d=4, min_s=1, feat_sub=None):
        self.n_trees = n_trees
        self.max_depth = max_d
        self.min_samples = min_s
        self.trees = []
        self.feature_subset = feat_sub #for RF
        self.oob_idx = []

    def bootstrap(self, X, y):
        n = X.shape[0]
        idx = np.random.choice(n, n, replace=True) #with replacement
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[idx] = False
        oob = np.where(oob_mask)[0]
        return X[idx], y[idx], oob

    def fit(self, X, y):
        self.trees = []
        self.oob_idx = []

        for _ in range(self.n_trees):
            X_b, y_b, oob = self.bootstrap(X, y)
            self.oob_idx.append(oob)

            tree = DecisionTree(max_d=self.max_depth, min_s=self.min_samples, feat_sub=self.feature_subset)
            tree.fit(X_b, y_b)
            self.trees.append(tree)

    def predict(self, X):
        final = []
        for i in range(X.shape[0]):
            preds = [tree.pred_sample(tree.root, X[i]) for tree in self.trees]
            pred = Counter(preds).most_common(1)[0][0]
            final.append(pred)
        return np.array(final)

    #oob_error
    def oob_error(self, X, y):
        n = X.shape[0]
        oob_preds = np.zeros((n, self.n_trees), dtype=int)
        oob_counts = np.zeros(n, dtype=int)

        for i, tree in enumerate(self.trees):
            oob = self.oob_idx[i]
            if len(oob) > 0:
                preds = tree.predict(X[oob])
                for j, idx in enumerate(oob):
                    oob_preds[idx, oob_counts[idx]] = preds[j]
                    oob_counts[idx] += 1

        final_preds = np.zeros(n, dtype=int)
        for i in range(n):
            if oob_counts[i] > 0:
                votes = oob_preds[i, :oob_counts[i]]
                final_preds[i] = Counter(votes).most_common(1)[0][0]

        valid = oob_counts > 0
        return np.sum(final_preds[valid] != y[valid]) / sum(valid)

# Data
data = [
    (25, "High", "No", "Fair", "No"),
    (30, "High", "No", "Excellent", "No"),
    (35, "Medium", "No", "Fair", "Yes"),
    (40, "Low", "No", "Fair", "Yes"),
    (45, "Low", "Yes", "Fair", "Yes"),
    (50, "Low", "Yes", "Excellent", "No"),
    (55, "Medium", "Yes", "Excellent", "Yes"),
    (60, "High", "No", "Fair", "No")
]

df = pd.DataFrame(data, columns=["Age", "Income", "Student", "Credit", "Buy"])
df["Buy"] = df["Buy"].map({"No": 0, "Yes": 1})
df["Income"] = df["Income"].map({"High": 0, "Medium": 1, "Low": 2})
df["Student"] = df["Student"].map({"No": 0, "Yes": 1})
df["Credit"] = df["Credit"].map({"Fair": 0, "Excellent": 1})

X = df.drop(columns=["Buy"]).values
y = df["Buy"].values

new_x = np.array([[42, 2, 0, 1]])

print("\n----- Q3: Single Decision Tree -----")
tree = DecisionTree(max_d=3, min_s=1)
tree.fit(X, y)
pred = tree.predict(new_x)
print("Single Tree Prediction:", "Yes" if pred[0] == 1 else "No")

print("\n----- Q4: Bagging with 10 trees -----")
bag = Bagging(n_trees=10, max_d=3, min_s=1)
bag.fit(X, y)
bag_pred = bag.predict(new_x)
print("Bagging Prediction:", "Yes" if bag_pred[0] == 1 else "No")

oob_err_bag = bag.oob_error(X, y)
print(f"Bagging OOB Error: {oob_err_bag:.4f}")

print("\n----- Q4: Random Forest with 2 random predictors -----")
rf = Bagging(n_trees=10, max_d=3, min_s=1, feat_sub=2)
rf.fit(X, y)
rf_pred = rf.predict(new_x)
print("Random Forest Prediction:", "Yes" if rf_pred[0] == 1 else "No")

oob_err_rf = rf.oob_error(X, y)
print(f"Random Forest OOB Error: {oob_err_rf:.4f}")
