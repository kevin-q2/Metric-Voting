import numpy as np
from metric_voting import GreedyMonroe
from metric_voting import uniform_profile


def test_basic_profile(basic_profile):
    E = GreedyMonroe()
    assert set(E.elect(basic_profile, 1).tolist()) == {0}
    
    samples = 1000
    two_winner = np.zeros((samples, 2))
    for i in range(samples):
        two_winner[i,:] = E.elect(basic_profile, 2)
        
    u, c = np.unique(two_winner, return_counts=True)
    assert len(u) == 3
    assert np.isclose(c[0]/samples, 0.5, atol=0.05)
    assert np.isclose(c[0]/samples, 0.5, atol=0.05)
    assert c[2] == samples


def test_num_winners():
    E = GreedyMonroe()
    for _ in range(10):
        profile = uniform_profile(200, 10)
        rand_k = np.random.choice([1,2,5])
        assert len(E.elect(profile, rand_k)) == rand_k
        
