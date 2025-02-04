import numpy as np
from metric_voting import ChamberlinCourant, Borda
from metric_voting import uniform_profile


def test_basic_profile(basic_profile):
    election = ChamberlinCourant()
    assert set(election.elect(basic_profile, 1).tolist()) == {0}
    assert set(election.elect(basic_profile, 2).tolist()) == {1,3}
    assert set(election.elect(basic_profile, 3).tolist()) == {1,2,3}
    assert set(election.elect(basic_profile, 4).tolist()) == {0, 1, 2, 3}

    
def test_permutation_profile(permutation_profile):
    election = ChamberlinCourant()    
    samples = 10000
    winners = np.zeros(samples, dtype = int)
    for i in range(samples):
        winner = election.elect(permutation_profile, 1)
        winners[i] = winner[0]
        
    _, counts = np.unique(winners, return_counts = True)
    assert len(counts) == 4
    assert np.allclose(counts[0]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[1]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[2]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[3]/samples, 0.25, atol = 0.05, rtol = 0)
    
    
def test_with_borda():
    election1 = ChamberlinCourant()
    election2 = Borda()
    profile = uniform_profile(200, 10)
    assert set(election1.elect(profile, 1).tolist()) == set(election2.elect(profile, 1).tolist()) 
    
    
def test_num_winners():
    election = ChamberlinCourant()
    for _ in range(10):
        profile = uniform_profile(200, 10)
        rand_k = np.random.randint(1, 10 + 1)
        assert len(election.elect(profile, rand_k)) == rand_k
        

def test_tiebreak(cc_tie_profile):
    election = ChamberlinCourant()
    
    samples = 1000
    winners = np.zeros(samples)
    for i in range(samples):
        winners[i] = election.elect(cc_tie_profile, 1)[0]
        
    unique, counts = np.unique(winners, return_counts=True)
    assert len(unique) == 2
    assert np.isclose(counts[0]/samples, 0.5, atol=0.05, rtol = 0)
    assert np.isclose(counts[1]/samples, 0.5, atol=0.05, rtol = 0)