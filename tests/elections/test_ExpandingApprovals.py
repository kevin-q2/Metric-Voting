import numpy as np
from metric_voting import ExpandingApprovals
from metric_voting import uniform_profile


def test_candidate_elect():
    E = ExpandingApprovals()
    n,m,k = 200, 100, 20
    quota = int(np.ceil(n/k))
    neighborhood = np.zeros((m, n))
    uncovered_mask = np.ones(n, dtype = bool)
    elected_mask = np.zeros(m, dtype = bool)
    random_elects = np.random.choice(range(m), 10, replace = False)
    
    for i, e in enumerate(random_elects):
        neighborhood[e, i*quota : (i + 1)*quota] = 1
        
    E.neighborhood = neighborhood
    E.uncovered_mask = uncovered_mask
    E.elected_mask = elected_mask
    E.quota = quota
    for e in random_elects:
        E.candidate_check_elect(e)
    
    assert set(np.where(E.elected_mask == 1)[0]) == set(random_elects)


def test_num_winners():
    E = ExpandingApprovals()
    for _ in range(100):
        profile = uniform_profile(200, 10)
        rand_k = np.random.randint(1, 10 + 1)
        assert len(E.elect(profile, rand_k)) == rand_k
        
        
def test_tie_break(expanding_fp_tie_profile):
    E = ExpandingApprovals()
    
    samples = 1000
    winners = np.zeros(samples, dtype = int)
    for i in range(samples):
        winner = E.elect(expanding_fp_tie_profile, 1)
        winners[i] = winner[0]
        
    _, counts = np.unique(winners, return_counts=True)
    assert len(counts) == 2
    assert np.allclose(counts[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts[1]/samples, 0.5, atol = 0.05, rtol = 0)
    
'''
NOTE: This does not work because of randomized ordering.
def test_agreement_profile(agreement_profile):
    E = ExpandingApprovals()
    assert set(E.elect(agreement_profile, 1).tolist()) == set([0])
    assert set(E.elect(agreement_profile, 2).tolist()) == set([0,1])
    assert set(E.elect(agreement_profile, 3).tolist()) == set([0,1,2])
    assert set(E.elect(agreement_profile, 4).tolist()) == set([0,1,2,3])
'''
 
def test_permutation_profile(permutation_profile):
    E = ExpandingApprovals()    
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
    