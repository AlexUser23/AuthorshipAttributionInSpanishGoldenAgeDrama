"""Small numerical checks for the scientific assumptions, without the corpus."""

from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from baselines import (Sample, fit_features, nearest_author, relative_frequencies,
                       select_svm, transform, word_counts, majority_author, split_samples)


def sample(**counts):
    return Sample(Counter(counts), "author", "play", 0)


class BaselineTests(unittest.TestCase):
    def test_tokenization_and_denominator(self):
        self.assertEqual(word_counts("¡ÉL, él! Y tú; 123 _ 9abc"),
                         Counter({"él": 2, "y": 1, "tú": 1}))
        assert_allclose(relative_frequencies([sample(a=1, outside=3)], ["a"]), [[0.25]])

    def test_training_only_statistics_and_constant_removal(self):
        train = [sample(a=1, b=1, c=2), sample(a=1, b=3, c=4)]
        x, fitted = fit_features(train, n_mfw=3)
        self.assertEqual(fitted[0], ["b", "a"])
        assert_allclose(x, [[-1, 1], [1, -1]])
        # Unseen words affect the denominator but never enter the vocabulary.
        assert_allclose(transform([sample(b=1, unseen=3)], fitted), [[-1, -3]])
        assert_allclose(fitted[1], [0.3125, 0.1875])
        assert_allclose(fitted[2], [0.0625, 0.0625])

    def test_mfw_ranking_uses_only_training_counts(self):
        _, fitted = fit_features([sample(b=3, a=1), sample(c=1, a=3)], n_mfw=2)
        self.assertEqual(fitted[0], ["a", "b"])
        transform([sample(unseen=1000)], fitted)
        self.assertEqual(fitted[0], ["a", "b"])

    def test_delta_metrics_choose_different_neighbors(self):
        train = np.array([[10., 0.], [1., 1.]])
        test = np.array([[1., 0.]])
        authors = np.array(["A", "B"])
        assert_array_equal(nearest_author(train, test, authors, "cityblock"), ["B"])
        assert_array_equal(nearest_author(train, test, authors, "cosine"), ["A"])
        assert_array_equal(nearest_author(np.array([[0., 1.], [2., 1.]]),
                                          np.array([[1., 1.]]), authors, "cityblock"), ["A"])
        with self.assertRaisesRegex(ValueError, "zero vector"):
            nearest_author(train, np.zeros((1, 2)), authors, "cosine")

    def test_validation_selects_c_and_breaks_ties(self):
        train = np.array([[-2.], [-1.], [1.], [2.]])
        authors = np.array(["A", "A", "B", "B"])
        # All three values tie on this separable validation set.
        model = select_svm(train, authors, np.array([[-3.], [3.]]), ["A", "B"])
        self.assertEqual(model.C, 0.1)
        # Controlled scores also check that the best candidate need not be first.
        with patch("baselines.accuracy_score", side_effect=[0.5, 1.0, 0.5]):
            model = select_svm(train, authors, np.array([[-3.], [3.]]), ["A", "B"])
        self.assertEqual(model.C, 1.0)

    def test_validation_excluded_from_training_and_test(self):
        samples = [Sample(Counter(a=1), "A", str(i), i) for i in range(12)]
        for fold in range(12):
            train, val, test = split_samples(samples, fold)
            self.assertEqual(len(train), 10)
            self.assertEqual([s.fold for s in val], [(fold + 1) % 12])
            self.assertEqual([s.fold for s in test], [fold])
            self.assertTrue({s.fold for s in train}.isdisjoint({fold, (fold + 1) % 12}))

    def test_majority_and_alphabetical_tie(self):
        self.assertEqual(majority_author(["B", "A", "B"]), "B")
        self.assertEqual(majority_author(["B", "A"]), "A")


if __name__ == "__main__":
    unittest.main()
