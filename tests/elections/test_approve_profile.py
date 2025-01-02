from metric_voting import Election
import numpy as np
import pytest

def test_basic_ranking():
    rank = np.arange(10)
    E = Election()
    assert E._is_complete_ranking(rank)
    
def test_incomplete_ranking():
    rank = np.array([0,1,2,3,6,7,8])
    E = Election()
    assert not E._is_complete_ranking(rank)

def test_ranking_with_repeats():
    rank = np.array([0,1,2,3,3,4,5,6,7,8,9])
    E = Election()
    assert not E._is_complete_ranking(rank)
    
def test_basic_profile(basic_profile):
    E = Election()
    E._approve_profile(basic_profile, k=1)
    
def test_incomplete_profile(incomplete_profile):
    E = Election()
    with pytest.raises(ValueError, match="Profile not in correct form."):
        E._approve_profile(incomplete_profile, k=1)
    
def test_profile_with_repeats(profile_with_repeats):
    E = Election()
    with pytest.raises(ValueError, match="Profile not in correct form."):
        E._approve_profile(profile_with_repeats, k=1)
