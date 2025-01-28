import numpy as np
from metric_voting import DMRD
from metric_voting import uniform_profile

def test_update_indices(simple_multiwinner_rd_profile):
    E = DMRD()
    E.n = 6
    E.profile = simple_multiwinner_rd_profile
    E.elected_mask = np.array([True, True, False])
    E.voter_indices = np.array([0, 0, 0, 0, 0, 0])
    E.update_indices()
    
    assert np.allclose(E.voter_indices, [2, 1, 2, 1, 0, 0])
    

def test_single_winner(expanding_fp_tie_profile):
    samples = 1000
    winners = np.zeros(samples)
    for i in range(samples):
        # rho factor should not matter for the single winner case. 
        winners[i] = DMRD(rho = 1/3).elect(expanding_fp_tie_profile, 1)[0]
        
    _, counts = np.unique(winners, return_counts=True)
    assert len(counts) == 2
    assert np.allclose(counts[0]/samples, 0.5, atol=0.05, rtol=0)
    assert np.allclose(counts[1]/samples, 0.5, atol=0.05, rtol=0)
    
    
def test_deterministic_multiwinner(expanding_fp_tie_profile):
    samples = 1000
    two_winners = np.zeros((samples, 2))
    three_winners = np.zeros((samples, 3))
    for i in range(samples):
        two_winners[i,:] = DMRD(rho = 1).elect(expanding_fp_tie_profile, 2)
        three_winners[i,:] = DMRD(rho = 1).elect(expanding_fp_tie_profile, 3)
        
    _, counts = np.unique(two_winners, return_counts=True)
    assert len(counts) == 2
    assert counts[0] == 1000
    assert counts[1] == 1000
    
    _, counts = np.unique(three_winners, return_counts=True)
    assert len(counts) == 3
    assert counts[0] == 1000
    assert counts[1] == 1000
    assert counts[2] == 1000
    
    
def test_simple_multiwinner(simple_multiwinner_rd_profile):
    samples = 10000
    two_winners = np.zeros((samples, 2))
    for i in range(samples):
        two_winners[i,:] = DMRD(rho = 1).elect(simple_multiwinner_rd_profile, 2)
        
    _, counts = np.unique(two_winners, return_counts=True)
    assert len(counts) == 3
    assert np.allclose(counts[0]/samples, 2/3, atol=0.05, rtol=0)
    assert np.allclose(counts[1]/samples, 2/3, atol=0.05, rtol=0)
    assert np.allclose(counts[2]/samples, 2/3, atol=0.05, rtol=0)
    

def test_simple_multiwinner_rho_half(simple_multiwinner_rd_profile):
    samples = 1000
    two_winners = np.zeros((samples, 2))
    for i in range(samples):
        two_winners[i,:] = DMRD(rho = 1/2).elect(simple_multiwinner_rd_profile, 2)
        
    _, counts = np.unique(two_winners, return_counts=True)
    assert len(counts) == 3
    assert np.allclose(counts[0]/samples, 2/3, atol=0.05, rtol=0)
    assert np.allclose(counts[1]/samples, 2/3, atol=0.05, rtol=0)
    assert np.allclose(counts[2]/samples, 2/3, atol=0.05, rtol=0)
    
    
def test_simple_multiwinner_rho_third(simple_multiwinner_rd_profile):
    p = np.array([1/18, 1/18, 1/6, 1/6, 1/6, 1/6])
    p /= np.sum(p)
    p1 = p[0]
    p2 = p[2]
    expected_prob = 2/6 + 2 * (2/6 * (2 * p2 + p1))
    
    samples = 1000
    two_winners = np.zeros((samples, 2))
    for i in range(samples):
        two_winners[i,:] = DMRD(rho = 1/3).elect(simple_multiwinner_rd_profile, 2)
        
    _, counts = np.unique(two_winners, return_counts=True)
    assert len(counts) == 3
    assert np.allclose(counts[0]/samples, expected_prob, atol=0.05, rtol=0)
    assert np.allclose(counts[1]/samples, expected_prob, atol=0.05, rtol=0)
    assert np.allclose(counts[2]/samples, expected_prob, atol=0.05, rtol=0)

    
def test_all_unique():
    n = 100
    m = 20
    k = 5
    E = DMRD(rho = 1/2)
    for _ in range(100):
        profile = uniform_profile(n, m)
        assert len(set(E.elect(profile, k))) == k