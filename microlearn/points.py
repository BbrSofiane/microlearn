"""
points.py — Patterns stored as instances in feature space.

k-Nearest Neighbors from scratch in pure Python.  The model IS the
training data.  Optimization happens at *prediction* time (searching
for nearest points), not at training time.  This is called "lazy
learning" and is unique to point-based ML.

Key concepts:
  - The model stores raw training points — no learned transformation.
  - Prediction = find k closest stored points, majority vote.
  - Distance metric (Euclidean) operates in the raw feature space.
  - No optimization during fit(); all work is deferred to predict().
  - Constrained to the raw feature space — no new learned features.
"""

import math


class KNN:
    """k-Nearest Neighbors: the lazy learner.

    >>> from microlearn.core import make_moons, train_test_split, accuracy
    >>> X, y = make_moons(200, noise=0.2)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    >>> model = KNN(k=5)
    >>> model.fit(X_tr, y_tr)
    >>> acc = accuracy(y_te, model.predict(X_te))
    >>> acc > 0.8
    True
    """

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    # ------------------------------------------------------------------
    # fit: store the data.  That's it.  This IS the model.
    # No optimisation happens here — this is "lazy learning".
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Store training data.  No optimisation — the data IS the model."""
        self.X_train = [list(row) for row in X]
        self.y_train = list(y)

    # ------------------------------------------------------------------
    # predict: optimisation happens HERE, at prediction time.
    # For each query we search all stored points for the k closest.
    # ------------------------------------------------------------------
    def predict(self, X):
        """Predict labels by majority vote of k nearest stored points."""
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x):
        # Compute Euclidean distance to every stored point
        distances = []
        for i, x_train in enumerate(self.X_train):
            d = _euclidean(x, x_train)
            distances.append((d, self.y_train[i]))

        # Sort by distance, pick k closest
        distances.sort(key=lambda pair: pair[0])
        k_nearest = distances[: self.k]

        # Majority vote
        votes = {}
        for _, label in k_nearest:
            votes[label] = votes.get(label, 0) + 1

        return max(votes, key=votes.get)

    # ------------------------------------------------------------------
    # Introspection: what does the model "know"?
    # ------------------------------------------------------------------
    def stored_points(self):
        """Return the stored training data — this IS the learned model."""
        return self.X_train, self.y_train

    def model_size(self):
        """Number of stored points.  For KNN, model size = training set size."""
        return len(self.X_train)

    def __repr__(self):
        n = len(self.X_train) if self.X_train else 0
        return f"KNN(k={self.k}, stored_points={n})"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _euclidean(a, b):
    """Euclidean distance between two points, from scratch."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
