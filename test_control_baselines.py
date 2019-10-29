"""
Python Unit tests for control_baselines.py classes and functions.
"""

import os
import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from control_baselines import BasicRandomSearch, AugmentedRandomSearch


def black_box_function(x): 
    """Simple, 2-dimensional function to test optimization algorithms.

    The maximum point is at x = [0, 1] where f(x) == 1.0. 
    """
    return -x[0] ** 2 - (x[1] - 1) ** 2 + 1


class TestGlobalSearchAlgorithms(unittest.TestCase):

    def test_BasicRandomSearch(self):

        brs = BasicRandomSearch([0.0, 0.0], black_box_function)
        brs.search(n_iter=50)
        params = brs.theta
        assert_almost_equal(params, [0,  1])
        self.assertAlmostEqual(brs.rollout(params), 1.0)


if __name__ == '__main__':

    unittest.main()