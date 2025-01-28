import numpy as np
from itertools import permutations
from metric_voting import PluralityVeto
from metric_voting import uniform_profile


def test_update_least_preffered():
    E = PluralityVeto()
    n, m, k = 200, 100, 50
    profile = uniform_profile(n, m)
    elims = np.random.choice(range(m), k, replace = False)
    eliminated = np.zeros(m)
    eliminated[elims] = 1
    
    E.profile = profile
    E.n, E.m = n, m
    E.eliminated = eliminated
    E.last_place_indices = np.zeros(n, dtype = int) + m - 1
    E.update_least_preffered()
    
    for i in range(n):
        last_place = E.last_place_indices[i]
        for j in range(m):
            if j == last_place:
                assert E.eliminated[profile[j,i]] == 0
            elif j > last_place:
                assert E.eliminated[profile[j,i]] == 1
                
                
def test_initialize(plurality_veto_elim_profile):
    E = PluralityVeto()
    E.profile = plurality_veto_elim_profile
    E.m, E.n = plurality_veto_elim_profile.shape
    E.k = 2
    E.candidate_scores = np.zeros(E.m)
    E.eliminated = np.zeros(E.m)
    E.last_place_indices = np.zeros(E.n, dtype=int) + E.m - 1
    E.initialize(2)
    
    #assert np.allclose(E.candidate_scores, [0, 2, 1, 3])
    assert np.allclose(E.candidate_scores, [0, 4, 3, 5])
    assert set(np.where(E.eliminated == 1)[0]) == {0}
    
    
def test_num_winners():
    E = PluralityVeto()
    for _ in range(100):
        profile = uniform_profile(200, 10)
        rand_k = np.random.randint(1, 10 + 1)
        assert len(E.elect(profile, rand_k)) == rand_k

def test_tie_break():
    E = PluralityVeto()
    profile = np.array(list(permutations([0,1,2,3]))).T
    
    samples = 1000
    winners = np.zeros(samples, dtype = int)
    for i in range(1000):
        winner = E.elect(profile, 1)
        winners[i] = winner[0]
        
    _, counts = np.unique(winners, return_counts = True)
    assert len(counts) == 4
    assert np.allclose(counts[0]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[1]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[2]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[3]/samples, 0.25, atol = 0.05, rtol = 0)
    
    
def test_agreement_profile(agreement_profile):
    E = PluralityVeto()
    assert set(E.elect(agreement_profile, 1).tolist()) == set([0])
    assert set(E.elect(agreement_profile, 2).tolist()) == set([0,1])
    assert set(E.elect(agreement_profile, 3).tolist()) == set([0,1,2])
    assert set(E.elect(agreement_profile, 4).tolist()) == set([0,1,2,3])
    
def test_permutation_profile(permutation_profile):
    E = PluralityVeto()    
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
        
    
    
    
            
    
    