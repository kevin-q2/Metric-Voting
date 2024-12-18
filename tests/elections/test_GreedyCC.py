import numpy as np
from metric_voting import GreedyCC, ChamberlinCourant, Borda
from metric_voting import uniform_profile


def test_basic_profile(basic_profile):
    E = GreedyCC()
    assert set(E.elect(basic_profile, 1).tolist()) == {0}
    assert set(E.elect(basic_profile, 2).tolist()) == {0, 3}
    assert set(E.elect(basic_profile, 3).tolist()) == {0, 1, 3}
    assert set(E.elect(basic_profile, 4).tolist()) == {0, 1, 2, 3}
    
    
def test_num_winners():
    E = GreedyCC()
    for _ in range(100):
        profile = uniform_profile(200, 10)
        rand_k = np.random.randint(1, 10 + 1)
        assert len(E.elect(profile, rand_k)) == rand_k
        
    
def test_fp_tie_break(profile_with_fp_borda_tie):
    E = GreedyCC()
    winners = np.zeros((1000, 1))

    for i in range(1000):
        winners[i,:] = E.elect(profile_with_fp_borda_tie, 1)

    _, counts = np.unique(winners, return_counts=True)

    assert  len(counts) == 2
    assert np.allclose(counts[0]/1000, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts[1]/1000, 0.5, atol = 0.05, rtol = 0)
        

def test_with_chamberlin():
    E1 = GreedyCC()
    E2 = ChamberlinCourant()
    
    for _ in range(10):
        profile = uniform_profile(200, 10)
        rand_k = np.random.randint(1, 10 + 1)
        E1_winners = E1.elect(profile, rand_k)
        obj1 = E1.objective
        E2_winners = E2.elect(profile, rand_k)
        obj2 = E2.objective
        
        assert obj1/obj2 >= 1 - 1/np.e
