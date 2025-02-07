import numpy as np
from metric_voting import ChamberlinCourant, ChamberlinCourantTiebreak, Borda
from metric_voting import uniform_profile


def test_basic_profile(basic_profile):
    election = ChamberlinCourantTiebreak()
    assert set(election.elect(basic_profile, 1).tolist()) == {0}
    assert set(election.elect(basic_profile, 2).tolist()) == {1,3}
    assert set(election.elect(basic_profile, 3).tolist()) == {1,2,3}
    assert set(election.elect(basic_profile, 4).tolist()) == {0, 1, 2, 3}

def test_with_borda():
    election1 = ChamberlinCourantTiebreak()
    election2 = Borda()
    profile = uniform_profile(100, 10)
    assert set(election1.elect(profile, 1).tolist()) == set(election2.elect(profile, 1).tolist()) 
    
    
def test_num_winners():
    election = ChamberlinCourantTiebreak()
    for _ in range(1000):
        profile = uniform_profile(100, 10)
        rand_k = np.random.randint(1, 10 + 1)
        assert len(election.elect(profile, rand_k)) == rand_k
        
        
def test_with_chamberlin():
    election1 = ChamberlinCourantTiebreak()
    election2 = ChamberlinCourant()
    for _ in range(10):
        profile = uniform_profile(100, 10)
        rand_k = np.random.randint(1, 10 + 1)
        election1.elect(profile, rand_k)
        election2.elect(profile, rand_k)
        assert np.isclose(election1.objective, election2.objective, atol = 1e-5, rtol = 0)
        
        
def test_tiebreak(cc_tie_profile):
    election = ChamberlinCourantTiebreak()
    
    samples = 1000
    winners = np.zeros(samples)
    for i in range(samples):
        winners[i] = election.elect(cc_tie_profile, 1)[0]
        
    unique, counts = np.unique(winners, return_counts=True)
    assert len(unique) == 2
    assert np.isclose(counts[0]/samples, 0.5, atol=0.05, rtol = 0)
    assert np.isclose(counts[1]/samples, 0.5, atol=0.05, rtol = 0)
    

def test_permutation_profile(permutation_profile):
    election = ChamberlinCourantTiebreak()    
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
    

def test_permutation_profile_with_last1(permutation_with_last1):
    election = ChamberlinCourantTiebreak()    
    samples = 1000
    winners = np.zeros(samples, dtype = int)
    for i in range(samples):
        winner = election.elect(permutation_with_last1, 1)
        winners[i] = winner[0]
        
    _, counts = np.unique(winners, return_counts = True)
    assert len(counts) == 3
    assert np.allclose(counts[0]/samples, 1/3, atol = 0.05, rtol = 0)
    assert np.allclose(counts[1]/samples, 1/3, atol = 0.05, rtol = 0)
    assert np.allclose(counts[2]/samples, 1/3, atol = 0.05, rtol = 0)
    
    for _ in range(samples):
        winners = election.elect(permutation_with_last1, 3)
        assert np.array_equal(winners, [0,1,2])


def test_permutation_profile_with_last1(permutation_with_last2):
    election = ChamberlinCourantTiebreak()    
    samples = 1000
    winners = np.zeros(samples, dtype = int)
    for i in range(samples):
        winner = election.elect(permutation_with_last2, 1)
        winners[i] = winner[0]
        
    _, counts = np.unique(winners, return_counts = True)
    assert len(counts) == 4
    assert np.allclose(counts[0]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[1]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[2]/samples, 0.25, atol = 0.05, rtol = 0)
    assert np.allclose(counts[3]/samples, 0.25, atol = 0.05, rtol = 0)
    
    for _ in range(samples):
        winners = election.elect(permutation_with_last2, 4)
        assert np.array_equal(winners, [0,1,2,3])
        
        
def test_permutation_profile_with_noise1(permutation_with_noise1):
    election = ChamberlinCourantTiebreak()    
    samples = 1000
    winners = np.zeros(samples, dtype = int)
    
    # Deterministic winner
    for i in range(samples):
        winner = election.elect(permutation_with_noise1, 1)
        winners[i] = winner[0]
        
    _, counts = np.unique(winners, return_counts = True)
    assert len(counts) == 1
    
    # Tie for second place between candidates 2 and 3
    winners = np.zeros(samples, dtype = int)
    for i in range(samples):
        winner = election.elect(permutation_with_noise1, 2)
        winners[i] = [c for c in winner if c != 1][0]
        
    _, counts = np.unique(winners, return_counts = True)
    assert len(counts) == 2
    assert np.allclose(counts[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts[1]/samples, 0.5, atol = 0.05, rtol = 0)
    
    
    for _ in range(samples):
        winners = election.elect(permutation_with_noise1, 3)
        assert np.array_equal(winners, [1,2,3])
        
        
def test_permutation_profile_with_noise2(permutation_with_noise2):
    election = ChamberlinCourantTiebreak()    
    samples = 1000
    winners = np.zeros(samples, dtype = int)
    
    # Tie for third place between candidates 0 and 3
    winners = np.zeros((samples, 3), dtype = int)
    for i in range(samples):
        winner = election.elect(permutation_with_noise2, 3)
        winners[i, :] = winner
        
    _, counts = np.unique(winners, return_counts = True)
    assert len(counts) == 4
    assert np.allclose(counts[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts[3]/samples, 0.5, atol = 0.05, rtol = 0)