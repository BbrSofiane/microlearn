"""
core.py — Shared foundation for microlearn.

Data generation, train/test splitting, evaluation metrics, and
decision-boundary plotting. All from scratch (numpy only for data
generation and plotting convenience).
"""

import math
import random

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_moons(n_samples=200, noise=0.2, seed=42):
    """Generate the two-moons binary classification dataset.

    Returns X (list of [x1, x2] pairs) and y (list of 0/1 labels).
    Pure Python + math — no sklearn.
    """
    rng = random.Random(seed)
    half = n_samples // 2
    X, y = [], []

    for i in range(half):
        angle = math.pi * i / half
        x1 = math.cos(angle) + rng.gauss(0, noise)
        x2 = math.sin(angle) + rng.gauss(0, noise)
        X.append([x1, x2])
        y.append(0)

    for i in range(half):
        angle = math.pi * i / half
        x1 = 1 - math.cos(angle) + rng.gauss(0, noise)
        x2 = 1 - math.sin(angle) - 0.5 + rng.gauss(0, noise)
        X.append([x1, x2])
        y.append(1)

    return X, y


def make_circles(n_samples=200, noise=0.1, factor=0.5, seed=42):
    """Generate concentric circles dataset."""
    rng = random.Random(seed)
    half = n_samples // 2
    X, y = [], []

    for i in range(half):
        angle = 2 * math.pi * i / half
        x1 = math.cos(angle) + rng.gauss(0, noise)
        x2 = math.sin(angle) + rng.gauss(0, noise)
        X.append([x1, x2])
        y.append(0)

    for i in range(half):
        angle = 2 * math.pi * i / half
        x1 = factor * math.cos(angle) + rng.gauss(0, noise)
        x2 = factor * math.sin(angle) + rng.gauss(0, noise)
        X.append([x1, x2])
        y.append(1)

    return X, y


def make_blobs(n_samples=200, centers=None, noise=0.4, seed=42):
    """Generate linearly separable blob clusters."""
    rng = random.Random(seed)
    if centers is None:
        centers = [[-1, -1], [1, 1]]
    per_center = n_samples // len(centers)
    X, y = [], []

    for label, (cx, cy) in enumerate(centers):
        for _ in range(per_center):
            X.append([cx + rng.gauss(0, noise), cy + rng.gauss(0, noise)])
            y.append(label)

    return X, y


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Shuffle and split data into train and test sets."""
    rng = random.Random(seed)
    indices = list(range(len(X)))
    rng.shuffle(indices)

    split = int(len(X) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)


def precision(y_true, y_pred, positive=1):
    """Precision: TP / (TP + FP)."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yp == positive and yt == positive)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yp == positive and yt != positive)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred, positive=1):
    """Recall: TP / (TP + FN)."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yp == positive and yt == positive)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yp != positive and yt == positive)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def confusion_matrix(y_true, y_pred):
    """Returns (TP, FP, FN, TN) for binary classification."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    return tp, fp, fn, tn


# ---------------------------------------------------------------------------
# Plotting helpers (requires matplotlib — only used in the notebook)
# ---------------------------------------------------------------------------

def plot_decision_boundary(model, X, y, ax=None, title=None, resolution=100):
    """Plot a model's decision boundary on 2D data.

    Works for any object with a .predict() method that accepts a list of
    [x1, x2] pairs and returns a list of 0/1 labels.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    X_np = np.array(X)
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = [[float(xx[i, j]), float(yy[i, j])]
            for i in range(resolution) for j in range(resolution)]
    preds = model.predict(grid)
    Z = np.array(preds).reshape(resolution, resolution)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 4))

    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#E0F0F1", "#FCE4EC"], alpha=0.7)
    ax.contour(xx, yy, Z, levels=[0.5], colors=["#01696F"], linewidths=1.5)

    y_np = np.array(y)
    for label, color, marker in [(0, "#01696F", "o"), (1, "#A84B2F", "s")]:
        mask = y_np == label
        ax.scatter(X_np[mask, 0], X_np[mask, 1], c=color, marker=marker,
                   edgecolors="white", s=30, linewidths=0.5)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return ax


def plot_all(models, X, y, titles=None):
    """Plot decision boundaries for multiple models side by side."""
    import matplotlib.pyplot as plt

    n = len(models)
    if titles is None:
        titles = [type(m).__name__ for m in models]

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, model, title in zip(axes, models, titles):
        plot_decision_boundary(model, X, y, ax=ax, title=title)

    plt.tight_layout()
    return fig
