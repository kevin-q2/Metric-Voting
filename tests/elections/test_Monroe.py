import numpy as np
from metric_voting import Monroe, ChamberlinCourant, Borda
from metric_voting import uniform_profile


def test_basic_profile(basic_monroe_profile):
    E = Monroe()
    assert set(E.elect(basic_monroe_profile, 1).tolist()) == {3}
    assert set(E.elect(basic_monroe_profile, 2).tolist()) == {1,3}
    assert set(E.elect(basic_monroe_profile, 3).tolist()) == {1,2,3}
    assert set(E.elect(basic_monroe_profile, 4).tolist()) == {0, 1, 2, 3}
    
    
def test_proportionality_constraints(monroe_vs_chamberlin_profile):
    E1 = Monroe()
    E2 = ChamberlinCourant()
    assert set(E1.elect(monroe_vs_chamberlin_profile, 2).tolist()) == {1,2}
    assert set(E2.elect(monroe_vs_chamberlin_profile, 2).tolist()) == {0,1}

def test_num_winners():
    E = Monroe()
    for _ in range(10):
        profile = uniform_profile(200, 10)
        rand_k = np.random.randint(1, 10 + 1)
        assert len(E.elect(profile, rand_k)) == rand_k