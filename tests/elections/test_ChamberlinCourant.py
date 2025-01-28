import numpy as np
from metric_voting import ChamberlinCourant, Borda
from metric_voting import uniform_profile


def test_basic_profile(basic_profile):
    E = ChamberlinCourant()
    assert set(E.elect(basic_profile, 1).tolist()) == {0}
    assert set(E.elect(basic_profile, 2).tolist()) == {1,3}
    assert set(E.elect(basic_profile, 3).tolist()) == {1,2,3}
    assert set(E.elect(basic_profile, 4).tolist()) == {0, 1, 2, 3}
    
'''
NOTE: This doesn't work and after thinking about it, I don't think it should,
    the second winner should be chosen randomly. 
def test_agreement_profile(agreement_profile):
    E = ChamberlinCourant()
    assert set(E.elect(agreement_profile, 1).tolist()) == set([0])
    assert set(E.elect(agreement_profile, 2).tolist()) == set([0,1])
    assert set(E.elect(agreement_profile, 3).tolist()) == set([0,1,2])
    assert set(E.elect(agreement_profile, 4).tolist()) == set([0,1,2,3])
''' 
    
def test_permutation_profile(permutation_profile):
    E = ChamberlinCourant()    
    samples = 1000
    winners = np.zeros(samples, dtype = int)
    for i in range(samples):
        winner = E.elect(permutation_profile, 1)
        winners[i] = winner[0]
        
    _, counts = np.unique(winners, return_counts = True)
    assert len(counts) == 4
    assert np.allclose(counts[0]/samples, 0.25, atol = 0.1, rtol = 0)
    assert np.allclose(counts[1]/samples, 0.25, atol = 0.1, rtol = 0)
    assert np.allclose(counts[2]/samples, 0.25, atol = 0.1, rtol = 0)
    assert np.allclose(counts[3]/samples, 0.25, atol = 0.1, rtol = 0)
    
    
def test_with_borda():
    E1 = ChamberlinCourant()
    E2 = Borda()
    profile = uniform_profile(200, 10)
    assert set(E1.elect(profile, 1).tolist()) == set(E2.elect(profile, 1).tolist()) 
    
    
def test_num_winners():
    E = ChamberlinCourant()
    for _ in range(10):
        profile = uniform_profile(200, 10)
        rand_k = np.random.randint(1, 10 + 1)
        assert len(E.elect(profile, rand_k)) == rand_k
        

def test_tiebreak(cc_tie_profile):
    E = ChamberlinCourant()
    
    samples = 1000
    winners = np.zeros(samples)
    for i in range(samples):
        winners[i] = E.elect(cc_tie_profile, 1)[0]
        
    unique, counts = np.unique(winners, return_counts=True)
    assert len(unique) == 2
    assert np.isclose(counts[0]/samples, 0.5, atol=0.05)
    assert np.isclose(counts[1]/samples, 0.5, atol=0.05)