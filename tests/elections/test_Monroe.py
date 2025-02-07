import numpy as np
from metric_voting import Monroe, ChamberlinCourant, Borda
from metric_voting import uniform_profile


def test_basic_profile(basic_monroe_profile):
    election = Monroe()
    assert set(election.elect(basic_monroe_profile, 1).tolist()) == {3}
    assert set(election.elect(basic_monroe_profile, 2).tolist()) == {1,3}
    assert set(election.elect(basic_monroe_profile, 3).tolist()) == {1,2,3}
    assert set(election.elect(basic_monroe_profile, 4).tolist()) == {0, 1, 2, 3}
    
def test_agreement_profile(agreement_profile):
    election = Monroe()
    assert set(election.elect(agreement_profile, 1).tolist()) == set([0])
    assert set(election.elect(agreement_profile, 2).tolist()) == set([0,1])
    assert set(election.elect(agreement_profile, 3).tolist()) == set([0,1,2])
    assert set(election.elect(agreement_profile, 4).tolist()) == set([0,1,2,3])
    
'''
def test_permutation_profile(permutation_profile):
    election = Monroe()    
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
''' 
    
def test_proportionality_constraints(monroe_vs_chamberlin_profile):
    election1 = Monroe()
    election2 = ChamberlinCourant()
    assert set(election1.elect(monroe_vs_chamberlin_profile, 2).tolist()) == {1,2}
    assert set(election2.elect(monroe_vs_chamberlin_profile, 2).tolist()) == {0,1}

def test_num_winners():
    election = Monroe()
    for _ in range(10):
        profile = uniform_profile(200, 10)
        rand_k = np.random.randint(1, 10 + 1)
        assert len(election.elect(profile, rand_k)) == rand_k
        
'''
def test_tiebreak(cc_tie_profile):
    election = Monroe()
    
    samples = 1000
    winners = np.zeros(samples)
    for i in range(samples):
        winners[i] = election.elect(cc_tie_profile, 1)[0]
        
    unique, counts = np.unique(winners, return_counts=True)
    assert len(unique) == 2
    assert np.isclose(counts[0]/samples, 0.5, atol=0.05)
    assert np.isclose(counts[1]/samples, 0.5, atol=0.05)
'''