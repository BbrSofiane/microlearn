# microlearn

A tiny library that implements the four learnable elements of ML from scratch, in ~750 lines of Python.

> "What are the building blocks of all the other machine learning models?"
> — Christoph Molnar, [*Elements of Machine Learning Algorithms*](https://mindfulmodeler.substack.com/p/points-rules-weights-distributions)

## The idea

[micrograd](https://github.com/karpathy/micrograd) teaches you what autograd *is* in ~150 lines. **microlearn** applies the same philosophy one level up: it teaches you that all ML models decompose into four learnable elements — **Points, Rules, Weights, and Distributions** — and implements one representative model for each, from scratch, so you can *see* how the same prediction task looks through four completely different lenses.

```
microlearn/
├── core.py              # Data generation, split, metrics, plotting
├── points.py            # k-NN from scratch (~90 lines)
├── rules.py             # Decision tree from scratch (~170 lines)
├── weights.py           # Logistic regression from scratch (~120 lines)
├── distributions.py     # Naive Bayes from scratch (~160 lines)
├── hybrids.py           # Model-based tree, rules + weights (~170 lines)
├── demo.ipynb           # Teaching notebook: all four on the same dataset
└── tests/
    └── test_elements.py # Tests comparing against scikit-learn
```

## Quick start

```bash
git clone https://github.com/BbrSofiane/microlearn.git
cd microlearn
pip install matplotlib  # only needed for the notebook
python -m pytest tests/ -v
```

```python
from microlearn.core import make_moons, train_test_split, accuracy, plot_all
from microlearn.points import KNN
from microlearn.rules import DecisionTree
from microlearn.weights import LogisticRegression
from microlearn.distributions import NaiveBayes

X, y = make_moons(n_samples=200, noise=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y)

models = [KNN(k=5), DecisionTree(max_depth=5), LogisticRegression(lr=0.5, epochs=500), NaiveBayes()]
for m in models:
    m.fit(X_train, y_train)
    print(f"{type(m).__name__:25s} accuracy={accuracy(y_test, m.predict(X_test)):.3f}")

plot_all(models, X_test, y_test)  # four decision boundaries, side by side
```

## The four elements

| Element | What it stores | How it predicts | How it optimises | Example models |
|---------|---------------|----------------|-----------------|----------------|
| **Points** | Training instances in feature space | Distance to nearest stored points | At prediction time (lazy learning) | k-NN, k-Means, RBF SVM |
| **Rules** | If-then conditions on features | Check which rules match, follow branches | Greedy combinatorial search | Decision trees, Random forests, XGBoost |
| **Weights** | Weight vector/tensor | Dot product with features + non-linearity | Gradient descent on differentiable loss | Logistic regression, CNNs, Transformers |
| **Distributions** | Distribution parameters (means, variances) | Bayes' theorem, conditional probability | Maximum likelihood estimation | Naive Bayes, Gaussian processes, Cox PH |

Each file implements one element from scratch in pure Python (no numpy in the model code). Every multiply, every comparison, every probability calculation is visible.

## What the notebook teaches

The [demo notebook](demo.ipynb) runs all four elements on the same dataset and walks through:

1. **Same data, four lenses** — side-by-side decision boundaries
2. **What gets stored** — training data vs. rules vs. weights vs. distribution parameters
3. **How optimisation differs** — lazy learning vs. greedy search vs. gradient descent vs. MLE
4. **How evaluation differs** — shared metrics vs. representation-specific criteria
5. **When each wins** — vary the dataset, watch which element fits which geometry
6. **The hybrid** — model-based tree (rules + weights) combines elements for richer models

## Design principles

Borrowed from [Karpathy's approach](https://github.com/karpathy/micrograd):

1. **Tiny** — each element fits in ~100-170 lines
2. **From scratch** — no sklearn, no PyTorch, no numpy in model code
3. **One file per element** — self-contained, readable top to bottom
4. **Unified interface** — every model has `fit(X, y)` / `predict(X)`
5. **Tests against reference** — verified against scikit-learn (same pattern as micrograd vs. PyTorch)

## Running tests

```bash
pip install pytest scikit-learn
python -m pytest tests/ -v
```

Tests compare predictions and accuracy against scikit-learn's implementations to verify correctness.

## Why this exists

micrograd teaches you what's inside ONE element (weights + gradient-based optimisation). microlearn zooms out and teaches you that weights are just one of four fundamental ways to store patterns — and that the choice between them shapes everything downstream: how you optimise, how you evaluate, and when the model wins or fails.

## References

- Christoph Molnar, ["Points, Rules, Weights, Distributions: The Elements of Machine Learning"](https://mindfulmodeler.substack.com/p/points-rules-weights-distributions)
- Andrej Karpathy, [micrograd](https://github.com/karpathy/micrograd)

## License

MIT
