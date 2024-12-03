from metric_voting import SNTV
import numpy as np

def test_basic_profile(basic_profile):
    E = SNTV()
    assert set(E.elect(basic_profile, 1).tolist()) == set([3])
    assert set(E.elect(basic_profile, 2).tolist()) == set([1,3])
    assert set(E.elect(basic_profile, 3).tolist()) == set([1,2,3])
    assert set(E.elect(basic_profile, 4).tolist()) == set([0,1,2,3])

def test_tie_break(profile_with_fp_tie):
    E = SNTV()
    winners = np.array([-1]*1000)

    for i in range(1000):
        winners[i] = E.elect(profile_with_fp_tie, 1)[0]

    _, counts = np.unique(winners, return_counts=True)

    assert  len(counts) == 2
    assert 450 < counts[0] and counts[0] < 550
    assert 450 < counts[1] and counts[1] < 550
