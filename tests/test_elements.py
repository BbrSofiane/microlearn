"""
test_elements.py — Verify microlearn against scikit-learn.

Same pattern as micrograd: test against a reference implementation
(scikit-learn) to confirm our from-scratch code is correct.
"""

import math
import pytest

from microlearn.core import make_moons, make_blobs, train_test_split, accuracy
from microlearn.points import KNN
from microlearn.rules import DecisionTree
from microlearn.weights import LogisticRegression
from microlearn.distributions import NaiveBayes
from microlearn.hybrids import ModelTree


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def moons_data():
    X, y = make_moons(300, noise=0.2, seed=42)
    return train_test_split(X, y, seed=42)


@pytest.fixture
def blobs_data():
    X, y = make_blobs(300, noise=0.3, seed=42)
    return train_test_split(X, y, seed=42)


# -----------------------------------------------------------------------
# KNN
# -----------------------------------------------------------------------

class TestKNN:
    def test_perfect_on_training_data_k1(self, moons_data):
        """With k=1, KNN should perfectly classify its own training data."""
        X_train, _, y_train, _ = moons_data
        model = KNN(k=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        assert accuracy(y_train, preds) == 1.0

    def test_reasonable_accuracy(self, moons_data):
        X_train, X_test, y_train, y_test = moons_data
        model = KNN(k=5)
        model.fit(X_train, y_train)
        acc = accuracy(y_test, model.predict(X_test))
        assert acc > 0.85

    def test_model_size_equals_training_size(self, moons_data):
        X_train, _, y_train, _ = moons_data
        model = KNN(k=5)
        model.fit(X_train, y_train)
        assert model.model_size() == len(X_train)

    def test_matches_sklearn(self, moons_data):
        """Compare predictions against sklearn's KNeighborsClassifier."""
        sklearn = pytest.importorskip("sklearn")
        from sklearn.neighbors import KNeighborsClassifier

        X_train, X_test, y_train, y_test = moons_data
        ours = KNN(k=5)
        ours.fit(X_train, y_train)
        our_preds = ours.predict(X_test)

        theirs = KNeighborsClassifier(n_neighbors=5)
        theirs.fit(X_train, y_train)
        their_preds = theirs.predict(X_test).tolist()

        agreement = sum(1 for a, b in zip(our_preds, their_preds) if a == b)
        assert agreement / len(our_preds) > 0.95


# -----------------------------------------------------------------------
# DecisionTree
# -----------------------------------------------------------------------

class TestDecisionTree:
    def test_perfect_on_linearly_separable(self, blobs_data):
        X_train, X_test, y_train, y_test = blobs_data
        model = DecisionTree(max_depth=3)
        model.fit(X_train, y_train)
        acc = accuracy(y_test, model.predict(X_test))
        assert acc > 0.95

    def test_reasonable_accuracy_moons(self, moons_data):
        X_train, X_test, y_train, y_test = moons_data
        model = DecisionTree(max_depth=5)
        model.fit(X_train, y_train)
        acc = accuracy(y_test, model.predict(X_test))
        assert acc > 0.85

    def test_depth_constraint(self, moons_data):
        X_train, _, y_train, _ = moons_data
        for max_depth in [1, 2, 3]:
            model = DecisionTree(max_depth=max_depth)
            model.fit(X_train, y_train)
            # Model should have at most 2^depth - 1 internal nodes
            assert model.model_size() <= 2 ** max_depth - 1

    def test_print_rules_runs(self, moons_data, capsys):
        X_train, _, y_train, _ = moons_data
        model = DecisionTree(max_depth=2)
        model.fit(X_train, y_train)
        model.print_rules()  # should not raise
        captured = capsys.readouterr()
        assert "feature[" in captured.out

    def test_matches_sklearn(self, moons_data):
        sklearn = pytest.importorskip("sklearn")
        from sklearn.tree import DecisionTreeClassifier

        X_train, X_test, y_train, y_test = moons_data
        ours = DecisionTree(max_depth=4)
        ours.fit(X_train, y_train)
        our_acc = accuracy(y_test, ours.predict(X_test))

        theirs = DecisionTreeClassifier(max_depth=4)
        theirs.fit(X_train, y_train)
        their_acc = theirs.score(X_test, y_test)

        # Our accuracy should be within 10% of sklearn's
        assert abs(our_acc - their_acc) < 0.10


# -----------------------------------------------------------------------
# LogisticRegression
# -----------------------------------------------------------------------

class TestLogisticRegression:
    def test_perfect_on_linearly_separable(self, blobs_data):
        X_train, X_test, y_train, y_test = blobs_data
        model = LogisticRegression(lr=0.5, epochs=500)
        model.fit(X_train, y_train)
        acc = accuracy(y_test, model.predict(X_test))
        assert acc > 0.95

    def test_loss_decreases(self, moons_data):
        X_train, _, y_train, _ = moons_data
        model = LogisticRegression(lr=0.5, epochs=500)
        model.fit(X_train, y_train)
        # Loss should generally decrease
        assert model.loss_history[-1] < model.loss_history[0]

    def test_model_size(self, moons_data):
        X_train, _, y_train, _ = moons_data
        model = LogisticRegression()
        model.fit(X_train, y_train)
        # 2 features + 1 bias
        assert model.model_size() == 3

    def test_predict_proba_range(self, moons_data):
        X_train, X_test, y_train, _ = moons_data
        model = LogisticRegression(lr=0.5, epochs=500)
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)
        assert all(0.0 <= p <= 1.0 for p in probas)

    def test_matches_sklearn(self, blobs_data):
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression as SkLR

        X_train, X_test, y_train, y_test = blobs_data
        ours = LogisticRegression(lr=0.5, epochs=1000)
        ours.fit(X_train, y_train)
        our_acc = accuracy(y_test, ours.predict(X_test))

        theirs = SkLR()
        theirs.fit(X_train, y_train)
        their_acc = theirs.score(X_test, y_test)

        assert abs(our_acc - their_acc) < 0.10


# -----------------------------------------------------------------------
# NaiveBayes
# -----------------------------------------------------------------------

class TestNaiveBayes:
    def test_perfect_on_linearly_separable(self, blobs_data):
        X_train, X_test, y_train, y_test = blobs_data
        model = NaiveBayes()
        model.fit(X_train, y_train)
        acc = accuracy(y_test, model.predict(X_test))
        assert acc > 0.95

    def test_stored_distributions_shape(self, moons_data):
        X_train, _, y_train, _ = moons_data
        model = NaiveBayes()
        model.fit(X_train, y_train)
        dists = model.stored_distributions()
        assert set(dists.keys()) == {0, 1}
        for c in [0, 1]:
            assert len(dists[c]["means"]) == 2
            assert len(dists[c]["variances"]) == 2
            assert 0 < dists[c]["prior"] < 1

    def test_log_likelihood_finite(self, moons_data):
        X_train, _, y_train, _ = moons_data
        model = NaiveBayes()
        model.fit(X_train, y_train)
        ll = model.log_likelihood(X_train, y_train)
        assert math.isfinite(ll)
        assert ll < 0  # log-likelihood of probabilities is always negative

    def test_matches_sklearn(self, blobs_data):
        sklearn = pytest.importorskip("sklearn")
        from sklearn.naive_bayes import GaussianNB

        X_train, X_test, y_train, y_test = blobs_data
        ours = NaiveBayes()
        ours.fit(X_train, y_train)
        our_preds = ours.predict(X_test)

        theirs = GaussianNB()
        theirs.fit(X_train, y_train)
        their_preds = theirs.predict(X_test).tolist()

        agreement = sum(1 for a, b in zip(our_preds, their_preds) if a == b)
        assert agreement / len(our_preds) > 0.95


# -----------------------------------------------------------------------
# ModelTree (Hybrid)
# -----------------------------------------------------------------------

class TestModelTree:
    def test_reasonable_accuracy(self, moons_data):
        X_train, X_test, y_train, y_test = moons_data
        model = ModelTree(max_depth=2, min_samples_leaf=10)
        model.fit(X_train, y_train)
        acc = accuracy(y_test, model.predict(X_test))
        assert acc > 0.75

    def test_works_on_blobs(self, blobs_data):
        X_train, X_test, y_train, y_test = blobs_data
        model = ModelTree(max_depth=2)
        model.fit(X_train, y_train)
        acc = accuracy(y_test, model.predict(X_test))
        assert acc > 0.90


# -----------------------------------------------------------------------
# Cross-element: same data, all models
# -----------------------------------------------------------------------

class TestAllElements:
    def test_all_models_predict_binary(self, moons_data):
        """All models should return only 0 or 1."""
        X_train, X_test, y_train, _ = moons_data
        models = [
            KNN(k=5),
            DecisionTree(max_depth=5),
            LogisticRegression(lr=0.5, epochs=300),
            NaiveBayes(),
            ModelTree(max_depth=2),
        ]
        for model in models:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            assert all(p in (0, 1) for p in preds), f"{type(model).__name__} gave non-binary"

    def test_all_models_beat_random(self, moons_data):
        """All models should beat random guessing (50%) by a clear margin."""
        X_train, X_test, y_train, y_test = moons_data
        models = [
            KNN(k=5),
            DecisionTree(max_depth=5),
            LogisticRegression(lr=0.5, epochs=500),
            NaiveBayes(),
            ModelTree(max_depth=2),
        ]
        for model in models:
            model.fit(X_train, y_train)
            acc = accuracy(y_test, model.predict(X_test))
            assert acc > 0.65, f"{type(model).__name__} only got {acc:.2f}"
