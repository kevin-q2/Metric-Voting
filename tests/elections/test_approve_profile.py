from metric_voting import Election
import numpy as np
import pytest

def test_basic_ranking():
    rank = np.arange(10)
    election = Election()
    assert election._is_complete_ranking(rank)
    
def test_incomplete_ranking():
    rank = np.array([0,1,2,3,6,7,8])
    election = Election()
    assert not election._is_complete_ranking(rank)

def test_ranking_with_repeats():
    rank = np.array([0,1,2,3,3,4,5,6,7,8,9])
    election = Election()
    assert not election._is_complete_ranking(rank)
    
def test_basic_profile(basic_profile):
    election = Election()
    election._approve_profile(basic_profile, k=1)
    
def test_incomplete_profile(incomplete_profile):
    election = Election()
    with pytest.raises(ValueError, match="Profile not in correct form."):
        election._approve_profile(incomplete_profile, k=1)
    
def test_profile_with_repeats(profile_with_repeats):
    election = Election()
    with pytest.raises(ValueError, match="Profile not in correct form."):
        election._approve_profile(profile_with_repeats, k=1)
