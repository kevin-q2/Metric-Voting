import pytest
import numpy as np

@pytest.fixture
def basic_profile():
    return np.array([
       [1, 3, 1, 3, 3, 0, 2, 3, 2, 1],
       [0, 0, 0, 0, 0, 1, 3, 0, 0, 0],
       [3, 2, 3, 2, 1, 3, 0, 2, 3, 3],
       [2, 1, 2, 1, 2, 2, 1, 1, 1, 2]
    ])

@pytest.fixture
def profile_with_fp_tie():
    return np.array([
       [1, 3, 1, 3, 3, 0, 1, 3, 2, 1],
       [0, 0, 0, 0, 0, 1, 3, 0, 0, 0],
       [3, 2, 3, 2, 1, 3, 0, 2, 3, 3],
       [2, 1, 2, 1, 2, 2, 2, 1, 1, 2]
    ])
