import numpy as np
from metric_voting.utils import *
from metric_voting.elections import Election



def test_euclidean_distance():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    assert euclidean_distance(x, y) == np.sqrt(27)
    
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    assert euclidean_distance(x, y) == 0
    
    
def test_tiebreak():
    # Test proxy tiebreak
    scores = np.array([1, 1, 1, 1, 1])
    proxy = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(tiebreak(scores, proxy), [0, 1, 2, 3, 4])
    
    # Test random tiebreak
    scores = np.array([1, 1, 1, 2, 2])
    proxy = np.array([1, 1, 2, 3, 4])
    samples = 1000
    lowest = np.zeros(samples)
    lowest_with_proxy = np.zeros(samples)
    for i in range(samples):
        lowest[i] = tiebreak(scores)[0]
        lowest_with_proxy[i] = tiebreak(scores, proxy)[0]
        
    _, counts = np.unique(lowest, return_counts=True)
    _, counts_with_proxy = np.unique(lowest_with_proxy, return_counts=True)
    
    assert len(counts) == 3
    assert np.allclose(counts[0]/samples, 1/3, atol = 0.05, rtol = 0)
    assert np.allclose(counts[1]/samples, 1/3, atol = 0.05, rtol = 0)
    assert np.allclose(counts[2]/samples, 1/3, atol = 0.05, rtol = 0)
    
    assert len(counts_with_proxy) == 2
    assert np.allclose(counts_with_proxy[0]/samples, 1/2, atol = 0.05, rtol = 0)
    assert np.allclose(counts_with_proxy[1]/samples, 1/2, atol = 0.05, rtol = 0)
    
    
    
def test_cost_array_to_ranking():
    cst_array = np.array([
        [4, 2, 4, 1],
        [1, 3, 1, 2],
        [3, 1, 3, 3],
        [2, 4, 2, 4]
    ])
    ranking = np.array([
        [1, 2, 1, 0],
        [3, 0, 3, 1],
        [2, 1, 2, 2],
        [0, 3, 0, 3]
    ])
    
    assert np.array_equal(cost_array_to_ranking(cst_array), ranking)
    
    cst_array_with_ties = np.array([
        [4, 2, 4, 1],
        [1, 3, 1, 1],
        [1, 1, 3, 3],
        [2, 4, 2, 4]
    ])
    
    samples = 1000
    fp0 = np.zeros(samples)
    fp3 = np.zeros(samples)
    for i in range(samples):
        ranking = cost_array_to_ranking(cst_array_with_ties)
        fp0[i] = ranking[0,0]
        fp3[i] = ranking[0,3]
        
    _, counts0 = np.unique(fp0, return_counts=True)
    _, counts3 = np.unique(fp3, return_counts=True)
    
    assert len(counts0) == 2
    assert np.allclose(counts0[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts0[1]/samples, 0.5, atol = 0.05, rtol = 0)
    
    assert len(counts3) == 2
    assert np.allclose(counts3[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts3[1]/samples, 0.5, atol = 0.05, rtol = 0)
        
        
        
def test_borda_matrix():
    profile = np.array([
        [1, 2, 1, 0],
        [3, 0, 3, 1],
        [2, 1, 2, 2],
        [0, 3, 0, 3]
    ])
    
    borda = np.array([
        [0, 2, 0, 3],
        [3, 1, 3, 2],
        [1, 3, 1, 1],
        [2, 0, 2, 0]
    ])
    
    assert np.array_equal(borda_matrix(profile, 2), borda)
    
    approval = lambda x, y, z: 1 if z <= y else 0
    
    approval_scores = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 0]
    ])
    
    assert np.array_equal(borda_matrix(profile, 2, approval), approval_scores)
    
    
def test_remove_candidates():
    profile = np.array([
        [1, 2, 1, 0],
        [3, 0, 3, 1],
        [2, 1, 2, 2],
        [0, 3, 0, 3]
    ])
    
    new_profile = np.array([
        [1, 0, 1, 0],
        [3, 1, 3, 1],
        [0, 3, 0, 3]
    ])
    
    assert np.array_equal(remove_candidates(profile, [2]), new_profile)
    
    # Remove candidates that are not there
    assert np.array_equal(remove_candidates(profile, [4,6,92]), profile)
    
    
    
def test_uniform_profile():
    m = 2
    n = 2
    samples = 1000
    fpv = np.zeros((samples, 2))
    for i in range(samples):
        profile = uniform_profile(n, m)
        fpv[i, :] = profile[0,:]
        
    assert np.isclose(np.mean(fpv[:,0]), 0.5, atol = 0.05)
    assert np.isclose(np.mean(fpv[:,1]), 0.5, atol = 0.05)
    
    
    m = 432
    n = 1036
    E = Election()
    profile = uniform_profile(n, m)
    E._approve_profile(profile, k=23)
    
    
def test_random_voter_bloc():
    n = 10
    k = 3
    t = 2
    dist1 = np.ones(n)/n
    
    samples = 1000
    voters1 = []
    for i in range(samples):
        bloc1 = random_voter_bloc(n, k, t, dist1)
        assert len(bloc1) >= 7
        assert len(bloc1) < 10
        
        voters1 += list(bloc1)
        
    _, counts1 = np.unique(voters1, return_counts=True)
    assert np.allclose(counts1/samples, dist1 * 8, atol = 0.05)
    
    
    