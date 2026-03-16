"""
weights.py — Patterns stored as learned weight tensors.

Logistic regression from scratch in pure Python.  Predictions are made
by multiplying a weight vector by the input features, passing through
a sigmoid, and thresholding.

Key concepts:
  - Patterns are stored as a weight vector + bias.
  - Prediction = dot(weights, features) + bias → sigmoid → threshold.
  - Gradient descent: directly optimise a differentiable loss function.
  - The training loss IS the evaluation metric — unique to weight-based ML.
  - Extends naturally to deep learning: stack layers, apply chain rule.
"""

import math


class LogisticRegression:
    """Logistic regression: gradient-based optimisation of weights.

    >>> from microlearn.core import make_moons, train_test_split, accuracy
    >>> X, y = make_moons(200, noise=0.2)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    >>> model = LogisticRegression(lr=0.5, epochs=500)
    >>> model.fit(X_tr, y_tr)
    >>> acc = accuracy(y_te, model.predict(X_te))
    >>> acc > 0.7
    True
    """

    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None       # learned weight vector
        self.b = 0.0        # learned bias
        self.loss_history = []  # track the loss curve for teaching

    # ------------------------------------------------------------------
    # fit: gradient descent — the optimisation strategy for weights.
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Train the model by gradient descent on binary cross-entropy."""
        n_samples = len(X)
        n_features = len(X[0])

        # Initialise weights to zero
        self.w = [0.0] * n_features
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            total_loss = 0.0

            for i in range(n_samples):
                # --- Forward pass: multiply weights by features ---
                z = _dot(self.w, X[i]) + self.b
                pred = _sigmoid(z)

                # --- Compute loss: binary cross-entropy ---
                # This IS the evaluation metric — unique to weight-based ML.
                eps = 1e-15
                loss = -(y[i] * math.log(pred + eps)
                         + (1 - y[i]) * math.log(1 - pred + eps))
                total_loss += loss

                # --- Backward pass: compute and apply gradients ---
                # The derivative of BCE w.r.t. z simplifies to (pred - y).
                # This is why weight-based ML loves differentiable losses.
                error = pred - y[i]

                for j in range(n_features):
                    self.w[j] -= self.lr * error * X[i][j] / n_samples
                self.b -= self.lr * error / n_samples

            self.loss_history.append(total_loss / n_samples)

    # ------------------------------------------------------------------
    # predict: apply stored weights to new data.
    # ------------------------------------------------------------------
    def predict(self, X):
        """Predict labels: dot(w, x) + b → sigmoid → threshold at 0.5."""
        return [self._predict_one(x) for x in X]

    def predict_proba(self, X):
        """Return raw probabilities (before thresholding)."""
        return [_sigmoid(_dot(self.w, x) + self.b) for x in X]

    def _predict_one(self, x):
        z = _dot(self.w, x) + self.b
        return 1 if _sigmoid(z) >= 0.5 else 0

    # ------------------------------------------------------------------
    # Introspection: what does the model "know"?
    # ------------------------------------------------------------------
    def stored_weights(self):
        """Return the learned weights and bias — this IS the model."""
        return self.w, self.b

    def model_size(self):
        """Number of learnable parameters (weights + bias)."""
        return len(self.w) + 1 if self.w else 0

    def __repr__(self):
        n = len(self.w) if self.w else 0
        return f"LogisticRegression(weights={n}, bias=1, lr={self.lr})"


# ----------------------------------------------------------------------
# Helpers — pure Python arithmetic
# ----------------------------------------------------------------------

def _dot(a, b):
    """Dot product of two vectors, from scratch."""
    return sum(ai * bi for ai, bi in zip(a, b))


def _sigmoid(z):
    """Sigmoid function: squashes any real number into (0, 1)."""
    # Clip to avoid overflow
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))
