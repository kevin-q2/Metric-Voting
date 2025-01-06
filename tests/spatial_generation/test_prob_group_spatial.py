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
        ProbabilisticGroupSpatial(
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
        ProbabilisticGroupSpatial(
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
        ProbabilisticGroupSpatial(
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
        ProbabilisticGroupSpatial(
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
        ProbabilisticGroupSpatial(
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
        ProbabilisticGroupSpatial(
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
        ProbabilisticGroupSpatial(
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
    
    generator = ProbabilisticGroupSpatial(
        n_voter_groups,
        n_cand_groups,
        voter_dists,
        voter_dist_params,
        cand_dists, 
        cand_dist_params
    )
    
    n = 3000
    m = 2000
    
    with pytest.raises(
        ValueError,
        match="Number of voter probabilities is inconsistent with n_voter_groups."
    ):
        generator.generate(n, m, [1/3], [3/4, 1/4])
        
    with pytest.raises(
        ValueError,
        match="Number of candidate probabilities is inconsistent with n_candidate_groups."
    ):
        generator.generate(n, m, [1/3, 2/3], [1/4])
    
    with pytest.raises(
        ValueError,
        match="Voter group probabilities do not sum to 1."
    ):
        generator.generate(n, m, [1/3, 1/3], [1/4, 3/4])
        
    with pytest.raises(
        ValueError,
        match="Candidate group probabilities do not sum to 1."
    ):
        generator.generate(n, m, [1/3, 2/3], [3/4, 3/4])
    
    
    voter_group_probs = [1/3, 2/3]
    cand_group_probs = [3/4, 1/4]
    
    (profile, 
    candidate_pos,
    voter_pos, 
    candidate_labels,
    voter_labels) = generator.generate(n, m, voter_group_probs, cand_group_probs)
    
    assert candidate_pos.shape == (m,2)
    assert voter_pos.shape == (n,2)
    
    cands_unique, cands_counts = np.unique(candidate_labels, return_counts = True)
    assert len(cands_unique) == n_cand_groups
    assert np.allclose(cands_counts/m, cand_group_probs, atol = 0.05)
    
    voters_unique, voters_counts = np.unique(voter_labels, return_counts = True)
    assert len(voters_unique) == n_voter_groups
    assert np.allclose(voters_counts/n, voter_group_probs, atol = 0.05)
    
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