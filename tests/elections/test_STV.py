import numpy as np
from metric_voting import STV
from metric_voting import uniform_profile
from votekit.elections import STV as votekit_STV
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.elections import fractional_transfer, random_transfer 


def test_fractional_basic_profile(basic_profile):
    stv_elector = STV(transfer_type = 'fractional')
    assert set(stv_elector(basic_profile, 1).tolist()) == set([3])
    assert set(stv_elector(basic_profile, 2).tolist()) == set([1,3])
    assert set(stv_elector(basic_profile, 4).tolist()) == set([0,1,2,3])
    
    # For this profile, 3rd place should be randomly decided between an elimination for
    # either candidate 0 or candidate 2.
    samples = 1000
    winners = np.zeros((samples, 3))
    for i in range(1000):
        winners[i,:] = stv_elector(basic_profile, 3)

    _, counts = np.unique(winners, return_counts=True)

    assert  len(counts) == 4
    assert 450 < counts[0] and counts[0] < 550
    assert 450 < counts[2] and counts[2] < 550
    
    assert set(stv_elector(basic_profile, 4).tolist()) == set([0,1,2,3])
    

def test_weighted_fractional_basic_profile(basic_profile):
    stv_elector = STV(transfer_type = 'weighted-fractional')
    assert set(stv_elector(basic_profile, 1).tolist()) == set([3])
    assert set(stv_elector(basic_profile, 2).tolist()) == set([1,3])
    assert set(stv_elector(basic_profile, 4).tolist()) == set([0,1,2,3])
    
    # For this profile, 3rd place should be randomly decided between an elimination for
    # either candidate 0 or candidate 2.
    samples = 1000
    winners = np.zeros((samples, 3))
    for i in range(1000):
        winners[i,:] = stv_elector(basic_profile, 3)

    _, counts = np.unique(winners, return_counts=True)

    assert  len(counts) == 4
    assert 450 < counts[0] and counts[0] < 550
    assert 450 < counts[2] and counts[2] < 550
    
    assert set(stv_elector(basic_profile, 4).tolist()) == set([0,1,2,3])
    
    
def test_cambridge_basic_profile(basic_profile):
    stv_elector = STV(transfer_type = 'cambridge')
    assert set(stv_elector(basic_profile, 1).tolist()) == set([3])
    assert set(stv_elector(basic_profile, 2).tolist()) == set([1,3])
    assert set(stv_elector(basic_profile, 4).tolist()) == set([0,1,2,3])
    
    # For this profile, 3rd place should be randomly decided between an elimination for
    # either candidate 0 or candidate 2.
    samples = 1000
    winners = np.zeros((samples, 3))
    for i in range(1000):
        winners[i,:] = stv_elector(basic_profile, 3)

    _, counts = np.unique(winners, return_counts=True)

    assert  len(counts) == 4
    assert 450 < counts[0] and counts[0] < 550
    assert 450 < counts[2] and counts[2] < 550
    
    assert set(stv_elector(basic_profile, 4).tolist()) == set([0,1,2,3])
    
    
def test_fractional_num_winners():
    stv_elector = STV(transfer_type = 'fractional')
    
    n = 20
    m = 10
    k = 5
    
    for _ in range(1000):
        profile = uniform_profile(n, m)
        winners = stv_elector(profile, k)
        assert len(winners) == k
        
        
def test_weighted_fractional_num_winners():
    stv_elector = STV(transfer_type = 'weighted-fractional')
    
    n = 20
    m = 10
    k = 5
    
    for _ in range(1000):
        profile = uniform_profile(n, m)
        winners = stv_elector(profile, k)
        assert len(winners) == k
        
        
def test_cambridge_num_winners():
    stv_elector = STV(transfer_type = 'cambridge')
    
    n = 20
    m = 10
    k = 5
    
    for _ in range(1000):
        profile = uniform_profile(n, m)
        winners = stv_elector(profile, k)
        assert len(winners) == k
        
        
def test_fractional_with_votekit():
    n = 20
    m = 10
    k = 5
    prof = uniform_profile(n, m)
    ballots = [Ballot(ranking = [{str(j)} for j in prof[:,i]], weight = 1) for i in range(n)]
    votekit_prof = PreferenceProfile(ballots= ballots)
    agree = 0 
    for _ in range(100):
        winners = STV(transfer_type = 'fractional')(prof, k)
        
        election = votekit_STV(votekit_prof, k, transfer = fractional_transfer)
        votekit_winners = election.get_elected()
        votekit_winners = [int(winner) for winner_set in
                            votekit_winners for winner in list(winner_set)]
        
        if set(winners) == set(votekit_winners):
            agree += 1
            
    assert agree > 50

    
    