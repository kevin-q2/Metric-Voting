from metric_voting import Bloc
import numpy as np

def test_basic_profile(basic_profile):
    E = Bloc()
    assert set(E.elect(basic_profile, 1).tolist()) == set([3])
    assert set(E.elect(basic_profile, 2).tolist()) == set([0,3])
    assert set(E.elect(basic_profile, 3).tolist()) == set([0,1,3])
    assert set(E.elect(basic_profile, 4).tolist()) == set([0,1,2,3])
    

def test_agreement_profile(agreement_profile):
    E = Bloc()
    assert set(E.elect(agreement_profile, 1).tolist()) == set([0])
    assert set(E.elect(agreement_profile, 2).tolist()) == set([0,1])
    assert set(E.elect(agreement_profile, 3).tolist()) == set([0,1,2])
    assert set(E.elect(agreement_profile, 4).tolist()) == set([0,1,2,3])
    
    
def test_permutation_profile(permutation_profile):
    E = Bloc()    
    samples = 1000
    winners = np.zeros(samples, dtype = int)
    for i in range(1000):
        winner = E.elect(permutation_profile, 1)
        winners[i] = winner[0]
        
    _, counts = np.unique(winners, return_counts = True)
    assert len(counts) == 4
    assert np.allclose(counts[0]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[1]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[2]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[3]/samples, 0.25, atol = 0.05, rtol = 0)


def test_fp_tie_break(profile_with_fp_tie):
    E = Bloc()
    winners = np.array([-1]*1000)

    for i in range(1000):
        winners[i] = E.elect(profile_with_fp_tie, 1)[0]

    _, counts = np.unique(winners, return_counts=True)

    assert  len(counts) == 2
    assert 450 < counts[0] and counts[0] < 550
    assert 450 < counts[1] and counts[1] < 550


def test_tie_break(profile_with_bloc_tie):
    E = Bloc()
    winners = np.zeros((1000, 3))

    for i in range(1000):
        winners[i,:] = E.elect(profile_with_bloc_tie, 3)

    _, counts = np.unique(winners, return_counts=True)

    assert  len(counts) == 4
    assert 450 < counts[1] and counts[1] < 550
    assert 450 < counts[2] and counts[2] < 550