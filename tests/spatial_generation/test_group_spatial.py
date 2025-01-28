import numpy as np
import pytest
from metric_voting.spatial_generation import *


def test_initialization():
    n_voter_groups = 2
    n_cand_groups = 2
    voter_dists = [np.random.normal, np.random.normal]
    voter_dist_params = [
        {"loc": [0,-1], "scale": 1, "size":2},
        {"loc": [0,1], "scale": 1, "size":2}
    ]
    cand_dists = [np.random.uniform, np.random.uniform]
    cand_dist_params = [
        {"low": 0, "high": 1, "size":2},
        {"low": 3, "high": 4, "size":2},
    ]
    
    with pytest.raises(
        ValueError,
        match="Group size does not match with given voter distributions."
    ):
        GroupSpatial(
            3,
            n_cand_groups,
            voter_dists,
            voter_dist_params,
            cand_dists, 
            cand_dist_params
        )
        
    with pytest.raises(
        ValueError,
        match="Group size does not match with given candidate distributions."
    ):
        GroupSpatial(
            n_voter_groups,
            3,
            voter_dists,
            voter_dist_params,
            cand_dists, 
            cand_dist_params
        )
        
    with pytest.raises(
        ValueError,
        match="No parameters were given for the input voter distribution."
    ):
        GroupSpatial(
            n_voter_groups,
            n_cand_groups,
            voter_dists,
            None,
            cand_dists, 
            cand_dist_params
        )
    
    with pytest.raises(
        ValueError,
        match="No parameters were given for the input candidate distribution."
    ):
        GroupSpatial(
            n_voter_groups,
            n_cand_groups,
            voter_dists,
            voter_dist_params,
            voter_dists, 
            None
        )
        
    with pytest.raises(
        TypeError,
        match="Invalid parameters for the voter distribution."
    ):
        GroupSpatial(
            n_voter_groups,
            n_cand_groups,
            voter_dists,
            cand_dist_params,
            cand_dists,
            cand_dist_params
        )
        
    with pytest.raises(
        TypeError,
        match="Invalid parameters for the candidate distribution."
    ):
        GroupSpatial(
            n_voter_groups,
            n_cand_groups,
            voter_dists,
            voter_dist_params,
            cand_dists,
            voter_dist_params
        )
        
        
    distance = lambda x,y,z : np.random.normal(x,y,z)
    
    with pytest.raises(
        ValueError,
        match=("Distance function is invalid or incompatible "
                "with voter/candidate distributions.")
    ):
        GroupSpatial(
            n_voter_groups,
            n_cand_groups,
            voter_dists,
            voter_dist_params,
            cand_dists,
            cand_dist_params,
            distance_fn = distance
        )
        
        

def test_generate():
    n_voter_groups = 2
    n_cand_groups = 2
    voter_dists = [np.random.normal, np.random.normal]
    voter_dist_params = [
        {"loc": [0,-1], "scale": 1, "size":2},
        {"loc": [0,1], "scale": 1, "size":2}
    ]
    cand_dists = [np.random.uniform, np.random.uniform]
    cand_dist_params = [
        {"low": 0, "high": 1, "size":2},
        {"low": 3, "high": 4, "size":2},
    ]
    
    generator = GroupSpatial(
        n_voter_groups,
        n_cand_groups,
        voter_dists,
        voter_dist_params,
        cand_dists, 
        cand_dist_params
    )
    
    voter_group_sizes = [1800, 1500]
    cand_group_sizes = [700, 600]
    
    (profile, 
    candidate_pos,
    voter_pos, 
    candidate_labels,
    voter_labels) = generator.generate(voter_group_sizes, cand_group_sizes)
    
    assert candidate_pos.shape == (np.sum(cand_group_sizes),2)
    assert voter_pos.shape == (np.sum(voter_group_sizes),2)
    
    assert len(np.unique(candidate_labels)) == n_cand_groups
    assert len(np.unique(voter_labels)) == n_voter_groups
    
    cand1 = np.where(candidate_labels == 0)[0]
    cand2 = np.where(candidate_labels == 1)[0]
    voter1 = np.where(voter_labels == 0)[0]
    voter2 = np.where(voter_labels == 1)[0]
    
    assert np.allclose(np.mean(candidate_pos[cand1,:], axis = 0), 0.5, atol = 0.05)
    assert np.max(candidate_pos[cand1,:]) <= 1
    assert np.min(candidate_pos[cand1,:]) >= 0
    
    assert np.allclose(np.mean(candidate_pos[cand2,:], axis = 0), 3.5, atol = 0.05)
    assert np.max(candidate_pos[cand2,:]) <= 4
    assert np.min(candidate_pos[cand2,:]) >= 3
    
    assert np.allclose(np.mean(voter_pos[voter1,:], axis = 0), [0,-1], atol = 0.05)
    assert np.allclose(np.std(voter_pos[voter1,:], axis = 0), 1, atol = 0.1)
    
    assert np.allclose(np.mean(voter_pos[voter2,:], axis = 0), [0,1], atol = 0.05)
    assert np.allclose(np.std(voter_pos[voter2,:], axis = 0), 1, atol = 0.1)