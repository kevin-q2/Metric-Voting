import numpy as np
from metric_voting import PAV
from metric_voting import uniform_profile

'''
def test_basic_profile(basic_profile):
    E = PAV()
    assert set(E.elect(basic_profile, 1).tolist()) == {3}
    assert E.objective == 4
    assert set(E.elect(basic_profile, 2).tolist()) == {0,3}
    assert E.objective == 12
    assert set(E.elect(basic_profile, 3).tolist()) == {0,1,3}
    assert E.objective == 17
    assert set(E.elect(basic_profile, 4).tolist()) == {0, 1, 2, 3}

def test_agreement_profile(agreement_profile):
    E = PAV()
    assert set(E.elect(agreement_profile, 1).tolist()) == set([0])
    assert set(E.elect(agreement_profile, 2).tolist()) == set([0,1])
    assert set(E.elect(agreement_profile, 3).tolist()) == set([0,1,2])
    assert set(E.elect(agreement_profile, 4).tolist()) == set([0,1,2,3])



# NOTE: The probabilities do not work out here... not sure why, need to revisit. 
def test_permutation_profile(permutation_profile):
    E = PAV()    
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

 
def test_num_winners():
    E = PAV()
    for _ in range(10):
        profile = uniform_profile(200, 10)
        rand_k = np.random.randint(1, 10 + 1)
        assert len(E.elect(profile, rand_k)) == rand_k
        
        
def test_tiebreak(cc_tie_profile):
    E = PAV()
    
    samples = 1000
    winners = np.zeros(samples)
    for i in range(samples):
        winners[i] = E.elect(cc_tie_profile, 1)[0]
        
    unique, counts = np.unique(winners, return_counts=True)
    assert len(unique) == 2
    assert np.isclose(counts[0]/samples, 0.5, atol=0.05)
    assert np.isclose(counts[1]/samples, 0.5, atol=0.05)
'''

'''
# code used for calculating objective.
b = borda_matrix(p, k = 3, scoring_scheme = lambda x, y, z: 1 if z <= y else 0)
zero_pos = np.array([np.where(p[:,i] == 0)[0][0] for i in range(10)])
one_pos = np.array([np.where(p[:,i] == 1)[0][0] for i in range(10)])
two_pos = np.array([np.where(p[:,i] == 2)[0][0] for i in range(10)])
three_pos = np.array([np.where(p[:,i] == 3)[0][0] for i in range(10)])

pos = np.vstack((zero_pos,one_pos,three_pos))
pos_rank = pos.argsort(axis = 0).argsort(axis = 0)
score = 1/(pos_rank + 1)
np.dot(b[0,:], score[0,:]) + np.dot(b[1,:], score[1,:]) + np.dot(b[3,:], score[2,:])
'''