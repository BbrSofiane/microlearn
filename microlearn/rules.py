"""
rules.py — Patterns stored as if-then conditions on features.

A decision tree from scratch in pure Python.  The tree learns rules of
the form ``feature_j <= threshold``, composed via recursive splitting.

Key concepts:
  - Rules are if-then conditions: feature <= threshold → go left, else right.
  - Combinatorial explosion: even 10 binary features → 3^10 = 59,049 rules.
  - Greedy optimisation: pick the SINGLE best split at each node, recurse.
  - Gini impurity measures how "mixed" the labels are at a node.
  - Works well for tabular data where features have clear semantics.
  - Coverage vs. accuracy trade-off: deeper trees cover fewer points per rule
    but with higher per-rule accuracy.
"""


class DecisionTree:
    """Decision tree: combinatorial optimisation via greedy recursion.

    >>> from microlearn.core import make_moons, train_test_split, accuracy
    >>> X, y = make_moons(200, noise=0.2)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    >>> model = DecisionTree(max_depth=5)
    >>> model.fit(X_tr, y_tr)
    >>> acc = accuracy(y_te, model.predict(X_te))
    >>> acc > 0.8
    True
    """

    def __init__(self, max_depth=5, min_samples_split=2, verbose=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.verbose = verbose
        self.tree = None

    # ------------------------------------------------------------------
    # fit: build the tree via recursive greedy splitting.
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Build the decision tree by recursively finding the best split."""
        self.n_features = len(X[0])
        self.tree = self._build_tree(X, y, depth=0)

    # ------------------------------------------------------------------
    # predict: walk the tree for each query point.
    # ------------------------------------------------------------------
    def predict(self, X):
        """Predict labels by walking the tree for each input."""
        return [self._predict_one(x, self.tree) for x in X]

    # ------------------------------------------------------------------
    # Tree construction — the greedy combinatorial search.
    # ------------------------------------------------------------------
    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        n_pos = sum(y)
        n_neg = n_samples - n_pos

        # Stopping conditions
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return _Leaf(self._majority(y), n_samples)
        if n_pos == 0 or n_neg == 0:
            return _Leaf(y[0], n_samples)

        # --- THE combinatorial challenge ---
        # Try every feature × every threshold to find the single best split.
        # This is where the "3^10 = 59,049 possible rules" problem lives.
        best = {"gini": float("inf"), "feature": None, "threshold": None}

        for feature_idx in range(self.n_features):
            values = sorted(set(row[feature_idx] for row in X))
            # Use midpoints between consecutive values as candidate thresholds
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                left_y, right_y = [], []
                for row, label in zip(X, y):
                    if row[feature_idx] <= threshold:
                        left_y.append(label)
                    else:
                        right_y.append(label)

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gini = _weighted_gini(left_y, right_y)
                if gini < best["gini"]:
                    best = {"gini": gini, "feature": feature_idx,
                            "threshold": threshold}

        # No useful split found
        if best["feature"] is None:
            return _Leaf(self._majority(y), n_samples)

        if self.verbose:
            print(f"{'  ' * depth}Split: feature[{best['feature']}] "
                  f"<= {best['threshold']:.3f}  (gini={best['gini']:.4f})")

        # Partition data and recurse — greedy, one rule at a time
        left_X, left_y, right_X, right_y = [], [], [], []
        for row, label in zip(X, y):
            if row[best["feature"]] <= best["threshold"]:
                left_X.append(row)
                left_y.append(label)
            else:
                right_X.append(row)
                right_y.append(label)

        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)

        return _Node(best["feature"], best["threshold"],
                     left_child, right_child, n_samples)

    def _predict_one(self, x, node):
        """Walk the tree: check which rule applies, follow the branch."""
        if isinstance(node, _Leaf):
            return node.label
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def _majority(self, y):
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        return max(counts, key=counts.get)

    # ------------------------------------------------------------------
    # Introspection: print the tree as human-readable rules.
    # ------------------------------------------------------------------
    def print_rules(self, node=None, depth=0, prefix=""):
        """Print the tree as human-readable if-then rules."""
        if node is None:
            node = self.tree
        indent = "  " * depth
        if isinstance(node, _Leaf):
            print(f"{indent}{prefix}→ class {node.label}  "
                  f"(covers {node.n_samples} points)")
            return
        print(f"{indent}{prefix}if feature[{node.feature}] "
              f"<= {node.threshold:.3f}:")
        self.print_rules(node.left, depth + 1, "yes: ")
        print(f"{indent}else:")
        self.print_rules(node.right, depth + 1, "no:  ")

    def model_size(self):
        """Count the number of rules (internal nodes) in the tree."""
        return self._count_nodes(self.tree)

    def _count_nodes(self, node):
        if isinstance(node, _Leaf):
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def __repr__(self):
        n = self.model_size() if self.tree else 0
        return f"DecisionTree(max_depth={self.max_depth}, rules={n})"


# ----------------------------------------------------------------------
# Tree node types
# ----------------------------------------------------------------------

class _Node:
    """Internal node: stores one rule (feature <= threshold)."""
    __slots__ = ("feature", "threshold", "left", "right", "n_samples")

    def __init__(self, feature, threshold, left, right, n_samples):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.n_samples = n_samples


class _Leaf:
    """Leaf node: stores a prediction (majority class label)."""
    __slots__ = ("label", "n_samples")

    def __init__(self, label, n_samples):
        self.label = label
        self.n_samples = n_samples


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _gini(y):
    """Gini impurity: probability that two randomly chosen samples disagree."""
    n = len(y)
    if n == 0:
        return 0.0
    counts = {}
    for label in y:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def _weighted_gini(left_y, right_y):
    """Weighted Gini of a split — the evaluation criterion for rules."""
    n = len(left_y) + len(right_y)
    return (len(left_y) / n) * _gini(left_y) + (len(right_y) / n) * _gini(right_y)
