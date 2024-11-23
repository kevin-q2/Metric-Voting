from metric_voting import is_complete_ranking, approve_profile
import numpy as np

def test_basic_ranking():
    rank = np.arange(10)
    assert is_complete_ranking(rank)
    
def test_incomplete_ranking():
    rank = np.array([0,1,2,3,6,7,8])
    assert not is_complete_ranking(rank)

def test_ranking_with_repeats():
    rank = np.array([0,1,2,3,3,4,5,6,7,8,9])
    assert not is_complete_ranking(rank)
    
def test_basic_profile(basic_profile):
    assert approve_profile(basic_profile)
    
def test_incomplete_profile(incomplete_profile):
    assert not approve_profile(incomplete_profile)
    
def test_profile_with_repeats(profile_with_repeats):
    assert not approve_profile(profile_with_repeats)
