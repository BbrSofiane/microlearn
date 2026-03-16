"""
distributions.py — Patterns stored as probability distributions.

Gaussian Naive Bayes from scratch in pure Python.  The model assumes
each feature follows a Gaussian distribution *per class* and stores the
distribution parameters (means, variances, priors).

Key concepts:
  - Distributional assumption is required: we choose Gaussian here.
  - Patterns are stored as distribution parameters (means, variances).
  - Prediction via Bayes' theorem: P(class|features) ∝ P(features|class) × P(class).
  - Optimisation = maximum likelihood estimation (MLE) — a closed-form solution.
  - Evaluation via log-likelihoods and likelihood ratios.
  - Some distributions collapse into weight-based representations:
    e.g. Gaussian target | linear features → linear regression.
"""

import math


class NaiveBayes:
    """Gaussian Naive Bayes: patterns stored as probability distributions.

    >>> from microlearn.core import make_moons, train_test_split, accuracy
    >>> X, y = make_moons(200, noise=0.2)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    >>> model = NaiveBayes()
    >>> model.fit(X_tr, y_tr)
    >>> acc = accuracy(y_te, model.predict(X_te))
    >>> acc > 0.7
    True
    """

    def __init__(self):
        self.classes = []
        self.means = {}       # {class: [mean_f0, mean_f1, ...]}
        self.variances = {}   # {class: [var_f0, var_f1, ...]}
        self.priors = {}      # {class: prior_probability}

    # ------------------------------------------------------------------
    # fit: maximum likelihood estimation — the optimisation for distributions.
    # No iteration, no gradients. Just compute the sufficient statistics.
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Fit by computing MLE of Gaussian parameters per class.

        This is the optimisation strategy for distribution-based ML:
        assume a distributional form, then find the parameters that
        maximise the likelihood of the observed data.  For Gaussians,
        MLE gives us the sample mean and variance — an exact solution.
        """
        self.classes = sorted(set(y))
        n_features = len(X[0])
        n_total = len(y)

        for c in self.classes:
            # Select rows belonging to this class
            X_c = [X[i] for i in range(len(y)) if y[i] == c]
            n_c = len(X_c)

            # MLE for Gaussian: mean = sample mean, variance = sample variance
            means = []
            variances = []
            for j in range(n_features):
                vals = [row[j] for row in X_c]
                mu = sum(vals) / n_c
                var = sum((v - mu) ** 2 for v in vals) / n_c
                # Add small epsilon to avoid division by zero
                var = max(var, 1e-9)
                means.append(mu)
                variances.append(var)

            self.means[c] = means
            self.variances[c] = variances
            self.priors[c] = n_c / n_total

    # ------------------------------------------------------------------
    # predict: apply Bayes' theorem using the stored distributions.
    # ------------------------------------------------------------------
    def predict(self, X):
        """Predict by computing posterior probabilities via Bayes' theorem."""
        return [self._predict_one(x) for x in X]

    def predict_log_proba(self, X):
        """Return log posterior probabilities for each class."""
        return [self._log_posteriors(x) for x in X]

    def _predict_one(self, x):
        posteriors = self._log_posteriors(x)
        return max(posteriors, key=posteriors.get)

    def _log_posteriors(self, x):
        """Compute log P(class|x) ∝ log P(x|class) + log P(class).

        We work in log-space to avoid numerical underflow from
        multiplying many small probabilities together.
        """
        log_posts = {}
        for c in self.classes:
            # Start with log prior: log P(class)
            log_post = math.log(self.priors[c])

            # Add log-likelihood of each feature under the class Gaussian:
            # log P(x_j | class) = log N(x_j; mean_j, var_j)
            for j in range(len(x)):
                log_post += _gaussian_log_pdf(
                    x[j], self.means[c][j], self.variances[c][j]
                )

            log_posts[c] = log_post
        return log_posts

    # ------------------------------------------------------------------
    # Evaluation: log-likelihood — the natural metric for distributions.
    # ------------------------------------------------------------------
    def log_likelihood(self, X, y):
        """Total log-likelihood of data under the model.

        This is the representation-specific evaluation criterion for
        distribution-based ML, analogous to how rules use coverage/accuracy
        and weights use the training loss.
        """
        ll = 0.0
        for x, label in zip(X, y):
            log_posts = self._log_posteriors(x)
            ll += log_posts[label]
        return ll

    # ------------------------------------------------------------------
    # Introspection: what does the model "know"?
    # ------------------------------------------------------------------
    def stored_distributions(self):
        """Return the stored distribution parameters — this IS the model."""
        return {
            c: {"means": self.means[c], "variances": self.variances[c],
                "prior": self.priors[c]}
            for c in self.classes
        }

    def model_size(self):
        """Number of stored parameters (means + variances + priors)."""
        n_features = len(self.means[self.classes[0]]) if self.classes else 0
        # Per class: n_features means + n_features variances + 1 prior
        return len(self.classes) * (2 * n_features + 1)

    def __repr__(self):
        return f"NaiveBayes(classes={self.classes}, params={self.model_size()})"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _gaussian_log_pdf(x, mean, var):
    """Log of the Gaussian probability density function, from scratch.

    log N(x; μ, σ²) = -0.5 * [log(2π) + log(σ²) + (x-μ)²/σ²]
    """
    return -0.5 * (math.log(2 * math.pi) + math.log(var) + (x - mean) ** 2 / var)
