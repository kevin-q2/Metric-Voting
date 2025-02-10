import numpy as np
import pytest
from metric_voting.spatial_generation import *


def test_initialization():
    voter_dist = np.random.normal
    voter_dist_params = {"loc": 0, "scale": 1, "size":2}
    cand_dist = np.random.uniform
    cand_dist_params = {"low": 0, "high": 1, "size":2}
    
    with pytest.raises(
        ValueError,
        match="No parameters were given for the input voter distribution."
    ):
        Spatial(voter_dist, None, cand_dist, cand_dist_params)
    
    with pytest.raises(
        ValueError,
        match="No parameters were given for the input candidate distribution."
    ):
        Spatial(voter_dist, voter_dist_params, voter_dist, None)
        
    with pytest.raises(
        TypeError,
        match="Invalid parameters for the voter distribution."
    ):
        Spatial(voter_dist, cand_dist_params, cand_dist, cand_dist_params)
        
    with pytest.raises(
        TypeError,
        match="Invalid parameters for the candidate distribution."
    ):
        Spatial(voter_dist, voter_dist_params, cand_dist, voter_dist_params)
        
        
    distance = lambda x,y,z : np.random.normal(x,y,z)
    
    with pytest.raises(
        ValueError,
        match=("Distance function is invalid or incompatible "
                "with voter/candidate distributions.")
    ):
        Spatial(voter_dist, voter_dist_params, cand_dist, cand_dist_params, distance_fn = distance)
        
        

def test_generate():
    voter_dist = np.random.normal
    voter_dist_params = {"loc": 0, "scale": 1, "size":2}
    cand_dist = np.random.uniform
    cand_dist_params = {"low": 0, "high": 1, "size":2}
    
    generator = Spatial(voter_dist, voter_dist_params, cand_dist, cand_dist_params)
    
    n = 1000
    m = 1000
    
    (profile, 
    candidate_pos,
    voter_pos, 
    candidate_labels,
    voter_labels) = generator.generate(n, m)
    
    assert candidate_pos.shape == (m,2)
    assert voter_pos.shape == (n,2)
    
    assert np.allclose(np.mean(candidate_pos, axis = 0), 0.5, atol = 0.05)
    assert np.max(candidate_pos) <= 1
    assert np.min(candidate_pos) >= 0
    
    assert np.allclose(np.mean(voter_pos, axis = 0), 0, atol = 0.05)
    assert np.allclose(np.std(voter_pos, axis = 0), 1, atol = 0.05)
    