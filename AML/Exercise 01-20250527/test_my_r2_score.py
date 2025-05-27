"""
Solution to exercise 01: TDD

DISCLAIMER:
Please note that this solution may contain errors (please report them, thanks!),
and that there are most-likely more elegant and more efficient implementations available
for the given problem. In this light, this solution may only serve as an inspiration for
solving the given exercise tasks.

(C) Merten Stender, TU Berlin
merten.stender@tu-berlin.de
"""

from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
import unittest
from my_r2_score import r2_score


# create some test cases by creating a test class
class TestR2Score(unittest.TestCase):
    # Requires the syntax test_ to recognize test

    # overview on test cases:
    # https://docs.python.org/3/library/unittest.html#test-cases

    def test_perfect_pred(self) -> None:
        # test the R2 score of a perfect prediction
        y_true = np.random.randn(100)
        y_pred = y_true
        self.assertAlmostEqual(r2_score(y_true, y_pred), 1.0)  # avoid rounding errors

    def test_mean_pred(self) -> None:
        # test the R2 score of a dummy model (must return 0, singularity!)
        y_true = np.random.randn(100)
        y_pred = np.ones_like(y_true) * np.mean(y_true)
        self.assertEqual(r2_score(y_true, y_pred), 0.0)  # avoid rounding errors

    def test_input_dims(self) -> None:
        # throw an error if dimensionalities dismatch
        y_true = np.random.randn(100)
        y_pred = np.random.randn(80)

        with self.assertRaises(ValueError):
            r2_score(y_true, y_pred)

    def test_data_type(self) -> None:
        # throw an error if dimensionalities dismatch
        y_true = np.random.randn(100)
        y_pred = list(np.random.randn(80))

        with self.assertRaises(TypeError):
            r2_score(y_true, y_pred)


if __name__ == "__main__":
    # use the main to run this script directly from your editor
    unittest.main()