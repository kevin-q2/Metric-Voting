from metric_voting import Bloc
import numpy as np

def test_basic_profile(basic_profile):
    assert set(Bloc(basic_profile, 1).tolist()) == set([3])
    assert set(Bloc(basic_profile, 2).tolist()) == set([0,3])
    assert set(Bloc(basic_profile, 3).tolist()) == set([0,1,3])
    assert set(Bloc(basic_profile, 4).tolist()) == set([0,1,2,3])

def test_fp_tie_break(profile_with_fp_tie):
    winners = np.array([-1]*1000)

    for i in range(1000):
        winners[i] = Bloc(profile_with_fp_tie, 1)[0]

    _, counts = np.unique(winners, return_counts=True)

    assert  len(counts) == 2
    assert 450 < counts[0] and counts[0] < 550
    assert 450 < counts[1] and counts[1] < 550


def test_bloc_tie_break(profile_with_bloc_tie):
    winners = np.zeros((1000, 3))

    for i in range(1000):
        winners[i,:] = Bloc(profile_with_bloc_tie, 3)

    _, counts = np.unique(winners, return_counts=True)

    assert  len(counts) == 4
    assert 450 < counts[1] and counts[1] < 550
    assert 450 < counts[2] and counts[2] < 550