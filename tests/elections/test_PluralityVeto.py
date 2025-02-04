import numpy as np
from itertools import permutations
from metric_voting import PluralityVeto
from metric_voting import uniform_profile


def test_update_least_preffered():
    election = PluralityVeto()
    n, m, k = 200, 100, 50
    profile = uniform_profile(n, m)
    elims = np.random.choice(range(m), k, replace = False)
    eliminated = np.zeros(m)
    eliminated[elims] = 1
    
    election.profile = profile
    election.n, election.m = n, m
    election.eliminated = eliminated
    election.last_place_indices = np.zeros(n, dtype = int) + m - 1
    election.update_least_preffered()
    
    for i in range(n):
        last_place = election.last_place_indices[i]
        for j in range(m):
            if j == last_place:
                assert election.eliminated[profile[j,i]] == 0
            elif j > last_place:
                assert election.eliminated[profile[j,i]] == 1
                
                
def test_initialize(plurality_veto_elim_profile):
    election = PluralityVeto()
    election.profile = plurality_veto_elim_profile
    election.m, election.n = plurality_veto_elim_profile.shape
    election.k = 2
    election.candidate_scores = np.zeros(election.m)
    election.eliminated = np.zeros(election.m)
    election.last_place_indices = np.zeros(election.n, dtype=int) + election.m - 1
    election.initialize(2)
    
    #assert np.allclose(election.candidate_scores, [0, 2, 1, 3])
    assert np.allclose(election.candidate_scores, [0, 4, 3, 5])
    assert set(np.where(election.eliminated == 1)[0]) == {0}
    
    
def test_num_winners():
    election = PluralityVeto()
    for _ in range(100):
        profile = uniform_profile(200, 10)
        rand_k = np.random.randint(1, 10 + 1)
        assert len(election.elect(profile, rand_k)) == rand_k

def test_tie_break():
    election = PluralityVeto()
    profile = np.array(list(permutations([0,1,2,3]))).T
    
    samples = 1000
    winners = np.zeros(samples, dtype = int)
    for i in range(1000):
        winner = election.elect(profile, 1)
        winners[i] = winner[0]
        
    _, counts = np.unique(winners, return_counts = True)
    assert len(counts) == 4
    assert np.allclose(counts[0]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[1]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[2]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[3]/samples, 0.25, atol = 0.05, rtol = 0)
    
    
def test_agreement_profile(agreement_profile):
    election = PluralityVeto()
    for _ in range(10):
        assert set(election.elect(agreement_profile, 1).tolist()) == set([0])
        assert set(election.elect(agreement_profile, 2).tolist()) == set([0,1])
        assert set(election.elect(agreement_profile, 3).tolist()) == set([0,1,2])
        assert set(election.elect(agreement_profile, 4).tolist()) == set([0,1,2,3])
    
def test_permutation_profile(permutation_profile):
    election = PluralityVeto()    
    samples = 1000
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
        
    
    
    
            
    
    