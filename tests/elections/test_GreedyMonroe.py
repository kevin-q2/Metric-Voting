import numpy as np
from metric_voting import GreedyMonroe, Monroe
from metric_voting import uniform_profile


def test_basic_profile(basic_profile):
    election = GreedyMonroe()
    assert set(election.elect(basic_profile, 1).tolist()) == {0}
    
    samples = 1000
    two_winner = np.zeros((samples, 2))
    for i in range(samples):
        two_winner[i,:] = election.elect(basic_profile, 2)
        
    u, c = np.unique(two_winner, return_counts=True)
    assert len(u) == 3
    assert np.isclose(c[0]/samples, 0.5, atol=0.05, rtol = 0)
    assert np.isclose(c[0]/samples, 0.5, atol=0.05, rtol = 0)
    assert c[2] == samples

def test_agreement_profile():
    election = GreedyMonroe()
    agreement_profile = np.array([
      [0,0,0,0],
      [1,1,1,1],
      [2,2,2,2],
      [3,3,3,3],
   ])
    assert set(election.elect(agreement_profile, 1).tolist()) == set([0])
    assert set(election.elect(agreement_profile, 2).tolist()) == set([0,1])
    assert set(election.elect(agreement_profile, 4).tolist()) == set([0,1,2,3])
    
def test_permutation_profile(permutation_profile):
    election = GreedyMonroe()    
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

def test_num_winners():
    election = GreedyMonroe()
    for _ in range(10):
        profile = uniform_profile(200, 10)
        rand_k = np.random.choice([1,2,5])
        assert len(election.elect(profile, rand_k)) == rand_k
        
        
def test_with_monroe():
    election1 = GreedyMonroe()
    election2 = Monroe()
    
    for _ in range(10):
        profile = uniform_profile(200, 10)
        rand_k = np.random.choice([1,2,5])
        election1_winners = election1.elect(profile, rand_k)
        obj1 = election1.objective
        election2_winners = election2.elect(profile, rand_k)
        obj2 = election2.objective
        
        assert obj1/obj2 >= 1 - 1/np.e
        
