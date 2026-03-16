"""
hybrids.py — Combining elements for richer models.

Model-based tree (Rules + Weights): a decision tree where each leaf
fits a logistic regression instead of returning a constant label.

Key concepts:
  - The four elements are not silos — combining them creates hybrids.
  - Rules partition the feature space into regions.
  - Weights fit a local linear model within each region.
  - The hybrid can learn non-linear boundaries that neither element
    can capture alone (rules handle the partitioning, weights handle
    the local gradients).
  - Other hybrids: GMMs (points + distributions), RBF SVM (points + weights).
"""

from microlearn.rules import DecisionTree, _Leaf, _Node
from microlearn.weights import LogisticRegression


class ModelTree:
    """Model-based tree: Rules + Weights hybrid.

    A decision tree that fits a logistic regression at each leaf instead
    of predicting the majority class.  The tree provides the partition
    (rules), and logistic regression provides the local decision boundary
    (weights).

    >>> from microlearn.core import make_moons, train_test_split, accuracy
    >>> X, y = make_moons(200, noise=0.2)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    >>> model = ModelTree(max_depth=2)
    >>> model.fit(X_tr, y_tr)
    >>> acc = accuracy(y_te, model.predict(X_te))
    >>> acc > 0.7
    True
    """

    def __init__(self, max_depth=3, min_samples_leaf=10):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.n_features = None

    def fit(self, X, y):
        """Build the tree, then fit a logistic regression at each leaf."""
        self.n_features = len(X[0])
        self.tree = self._build(X, y, depth=0)

    def predict(self, X):
        """Predict by walking the tree to a leaf, then using its local model."""
        return [self._predict_one(x, self.tree) for x in X]

    # ------------------------------------------------------------------
    # Build: split with rules, fit weights at leaves.
    # ------------------------------------------------------------------
    def _build(self, X, y, depth):
        n_samples = len(y)
        n_pos = sum(y)
        n_neg = n_samples - n_pos

        # At leaf: fit a logistic regression (WEIGHTS) instead of majority vote
        if (depth >= self.max_depth
                or n_samples < self.min_samples_leaf
                or n_pos == 0 or n_neg == 0):
            return self._make_leaf(X, y)

        # Find the best split using RULES logic (same as DecisionTree)
        best = {"gini": float("inf"), "feature": None, "threshold": None}

        for feature_idx in range(self.n_features):
            values = sorted(set(row[feature_idx] for row in X))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                left_y = [y[j] for j in range(n_samples) if X[j][feature_idx] <= threshold]
                right_y = [y[j] for j in range(n_samples) if X[j][feature_idx] > threshold]

                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue

                gini = _weighted_gini(left_y, right_y)
                if gini < best["gini"]:
                    best = {"gini": gini, "feature": feature_idx,
                            "threshold": threshold}

        if best["feature"] is None:
            return self._make_leaf(X, y)

        left_X, left_y, right_X, right_y = [], [], [], []
        for row, label in zip(X, y):
            if row[best["feature"]] <= best["threshold"]:
                left_X.append(row)
                left_y.append(label)
            else:
                right_X.append(row)
                right_y.append(label)

        left_child = self._build(left_X, left_y, depth + 1)
        right_child = self._build(right_X, right_y, depth + 1)

        return _HybridNode(best["feature"], best["threshold"],
                           left_child, right_child)

    def _make_leaf(self, X, y):
        """Create a leaf with a local logistic regression model."""
        n_pos = sum(y)
        n_neg = len(y) - n_pos

        # If pure or too few samples, fall back to majority vote
        if n_pos == 0 or n_neg == 0 or len(y) < 4:
            label = 1 if n_pos >= n_neg else 0
            return _HybridLeaf(label=label, local_model=None)

        # Fit a logistic regression on this leaf's data
        lr = LogisticRegression(lr=0.5, epochs=200)
        lr.fit(X, y)
        return _HybridLeaf(label=None, local_model=lr)

    def _predict_one(self, x, node):
        if isinstance(node, _HybridLeaf):
            if node.local_model is not None:
                return node.local_model._predict_one(x)
            return node.label
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def __repr__(self):
        return f"ModelTree(max_depth={self.max_depth})"


# ----------------------------------------------------------------------
# Node types for the hybrid tree
# ----------------------------------------------------------------------

class _HybridNode:
    """Internal node (a rule): feature <= threshold."""
    __slots__ = ("feature", "threshold", "left", "right")

    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class _HybridLeaf:
    """Leaf node: contains a local logistic regression (or a fallback label)."""
    __slots__ = ("label", "local_model")

    def __init__(self, label, local_model):
        self.label = label
        self.local_model = local_model


# ----------------------------------------------------------------------
# Reuse gini helper
# ----------------------------------------------------------------------

def _gini(y):
    n = len(y)
    if n == 0:
        return 0.0
    counts = {}
    for label in y:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def _weighted_gini(left_y, right_y):
    n = len(left_y) + len(right_y)
    return (len(left_y) / n) * _gini(left_y) + (len(right_y) / n) * _gini(right_y)
