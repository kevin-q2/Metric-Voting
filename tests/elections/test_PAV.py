import numpy as np
from metric_voting import PAV
from metric_voting import uniform_profile


def test_basic_profile(basic_profile):
    E = PAV()
    assert set(E.elect(basic_profile, 1).tolist()) == {3}
    assert E.objective == 4
    assert set(E.elect(basic_profile, 2).tolist()) == {0,3}
    assert E.objective == 12
    assert set(E.elect(basic_profile, 3).tolist()) == {0,1,3}
    assert E.objective == 17
    assert set(E.elect(basic_profile, 4).tolist()) == {0, 1, 2, 3}
    
    
def test_num_winners():
    E = PAV()
    for _ in range(10):
        profile = uniform_profile(200, 10)
        rand_k = np.random.randint(1, 10 + 1)
        assert len(E.elect(profile, rand_k)) == rand_k
        
        
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