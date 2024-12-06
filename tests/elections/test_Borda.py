from metric_voting import Borda
import numpy as np

def test_basic_profile(basic_profile):
    E = Borda()
    assert set(E.elect(basic_profile, 1).tolist()) == set([0])
    assert set(E.elect(basic_profile, 2).tolist()) == set([0,3])
    assert set(E.elect(basic_profile, 3).tolist()) == set([0,1,3])
    assert set(E.elect(basic_profile, 4).tolist()) == set([0,1,2,3])


def test_fp_tie_break(profile_with_fp_borda_tie):
    E = Borda()
    winners = np.zeros((1000, 1))

    for i in range(1000):
        winners[i,:] = E.elect(profile_with_fp_borda_tie, 1)

    _, counts = np.unique(winners, return_counts=True)

    assert  len(counts) == 2
    assert np.allclose(counts[0]/1000, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts[1]/1000, 0.5, atol = 0.05, rtol = 0)
    
    
def test_tie_break(profile_with_full_borda_tie):
    E = Borda()
    winners = np.zeros((1000, 2))

    for i in range(1000):
        winners[i,:] = E.elect(profile_with_full_borda_tie, 2)

    _, counts = np.unique(winners, return_counts=True)

    assert  len(counts) == 4
    assert np.allclose(counts[0]/1000, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts[1]/1000, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts[2]/1000, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts[3]/1000, 0.5, atol = 0.05, rtol = 0)