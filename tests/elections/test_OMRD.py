import numpy as np
from metric_voting import OMRD
from metric_voting import uniform_profile


def test_single_winner(expanding_fp_tie_profile):
    samples = 1000
    winners = np.zeros(samples)
    for i in range(samples):
        winners[i] = OMRD().elect(expanding_fp_tie_profile, 1)[0]
        
    _, counts = np.unique(winners, return_counts=True)
    assert len(counts) == 2
    assert np.allclose(counts[0]/samples, 0.5, atol=0.05, rtol=0)
    assert np.allclose(counts[1]/samples, 0.5, atol=0.05, rtol=0)
    
    
def test_deterministic_multiwinner(expanding_fp_tie_profile):
    samples = 1000
    two_winners = np.zeros((samples, 2))
    three_winners = np.zeros((samples, 3))
    for i in range(samples):
        two_winners[i,:] = OMRD().elect(expanding_fp_tie_profile, 2)
        three_winners[i,:] = OMRD().elect(expanding_fp_tie_profile, 3)
        
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
        two_winners[i,:] = OMRD().elect(simple_multiwinner_rd_profile, 2)
        
    _, counts = np.unique(two_winners, return_counts=True)
    assert len(counts) == 3
    assert np.allclose(counts[0]/samples, 2/3, atol=0.05, rtol=0)
    assert np.allclose(counts[1]/samples, 2/3, atol=0.05, rtol=0)
    assert np.allclose(counts[2]/samples, 2/3, atol=0.05, rtol=0)
    
    
def test_next_available(two_winner_rd_profile):
    samples = 1000
    winners = np.zeros((samples, 2))
    for i in range(samples):
        winners[i] = OMRD().elect(two_winner_rd_profile, 2)
        
    _, counts = np.unique(winners, return_counts=True)
    assert len(counts) == 3
    assert counts[0] == 1000
    assert np.allclose(counts[1]/samples, 0.5, atol=0.05, rtol=0)
    assert np.allclose(counts[2]/samples, 0.5, atol=0.05, rtol=0)
    
    
def test_all_unique():
    n = 100
    m = 20
    k = 5
    E = OMRD()
    for _ in range(100):
        profile = uniform_profile(n, m)
        assert len(set(E.elect(profile, k))) == k