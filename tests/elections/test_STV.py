import sys
import contextlib
import itertools
import numpy as np
from metric_voting import STV
from metric_voting import uniform_profile
from votekit.elections import STV as votekit_STV
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.elections import fractional_transfer, random_transfer 


def test_fractional_basic_profile(basic_profile):
    stv = STV(transfer_type = 'fractional')
    assert set(stv.elect(basic_profile, 1).tolist()) == set([3])
    assert set(stv.elect(basic_profile, 2).tolist()) == set([1,3])
    assert set(stv.elect(basic_profile, 4).tolist()) == set([0,1,2,3])
    
    # For this profile, 3rd place should be randomly tiebroken between an elimination for
    # either candidate 0 or candidate 2.
    # NOTE: This assumes fully random tiebreaks
    stv = STV(transfer_type = 'fractional', tiebreak_type = 'random')
    samples = 1000
    winners = np.zeros((samples, 3))
    for i in range(samples):
        winners[i,:] = stv.elect(basic_profile, 3)

    _, counts = np.unique(winners, return_counts=True)

    assert  len(counts) == 4
    assert np.allclose(counts[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts[2]/samples, 0.5, atol = 0.05, rtol = 0)
    
    # NOTE: This assumes tiebreaks first decided by first place votes, then random.
    stv = STV(transfer_type = 'weighted-fractional', tiebreak_type = 'fpv_random')
    samples = 1000
    winners = np.zeros((samples, 3))
    for i in range(samples):
        winners[i,:] = stv.elect(basic_profile, 3)

    unique_winners, counts = np.unique(winners, return_counts=True)
    assert set(unique_winners) == {1,2,3}
    

def test_weighted_fractional_basic_profile(basic_profile):
    stv = STV(transfer_type = 'weighted-fractional')
    assert set(stv.elect(basic_profile, 1).tolist()) == set([3])
    assert set(stv.elect(basic_profile, 2).tolist()) == set([1,3])
    assert set(stv.elect(basic_profile, 4).tolist()) == set([0,1,2,3])
    
    # For this profile, 3rd place should be tiebroken between an elimination for
    # either candidate 0 or candidate 2.
    # NOTE: This assumes fully random tiebreaks
    stv = STV(transfer_type = 'weighted-fractional', tiebreak_type = 'random')
    samples = 1000
    winners = np.zeros((samples, 3))
    for i in range(samples):
        winners[i,:] = stv.elect(basic_profile, 3)

    _, counts = np.unique(winners, return_counts=True)

    assert  len(counts) == 4
    assert np.allclose(counts[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts[2]/samples, 0.5, atol = 0.05, rtol = 0)
    
    # NOTE: This assumes tiebreaks first decided by first place votes, then random.
    stv = STV(transfer_type = 'weighted-fractional', tiebreak_type = 'fpv_random')
    samples = 1000
    winners = np.zeros((samples, 3))
    for i in range(samples):
        winners[i,:] = stv.elect(basic_profile, 3)

    unique_winners, counts = np.unique(winners, return_counts=True)
    assert set(unique_winners) == {1,2,3}
    
    
def test_cambridge_basic_profile(basic_profile):
    stv = STV(transfer_type = 'cambridge')
    assert set(stv.elect(basic_profile, 1).tolist()) == set([3])
    assert set(stv.elect(basic_profile, 2).tolist()) == set([1,3])
    assert set(stv.elect(basic_profile, 4).tolist()) == set([0,1,2,3])
    
    # For this profile, 3rd place should be tiebroken between an elimination for
    # either candidate 0 or candidate 2.
    
    # NOTE: This assumes fully random tiebreaks
    stv = STV(transfer_type = 'cambridge', tiebreak_type = 'random')
    samples = 1000
    winners = np.zeros((samples, 3))
    for i in range(samples):
        winners[i,:] = stv.elect(basic_profile, 3)

    _, counts = np.unique(winners, return_counts=True)

    assert  len(counts) == 4
    assert np.allclose(counts[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts[2]/samples, 0.5, atol = 0.05, rtol = 0)
    
    
    # NOTE: This assumes tiebreaks first decided by first place votes, then random.
    stv = STV(transfer_type = 'cambridge', tiebreak_type = 'fpv_random')
    samples = 1000
    winners = np.zeros((samples, 3))
    for i in range(samples):
        winners[i,:] = stv.elect(basic_profile, 3)

    unique_winners, counts = np.unique(winners, return_counts=True)
    assert set(unique_winners) == {1,2,3}

    
    
def test_fractional_num_winners():
    stv = STV(transfer_type = 'fractional')
    n = 20
    m = 10
    k = 5
    samples = 1000
    for _ in range(samples):
        profile = uniform_profile(n, m)
        winners = stv.elect(profile, k)
        assert len(winners) == k
        
        
def test_weighted_fractional_num_winners():
    stv = STV(transfer_type = 'weighted-fractional')
    n = 20
    m = 10
    k = 5
    samples = 1000
    for _ in range(samples):
        profile = uniform_profile(n, m)
        winners = stv.elect(profile, k)
        assert len(winners) == k
        
        
def test_cambridge_num_winners():
    stv = STV(transfer_type = 'cambridge')
    n = 20
    m = 10
    k = 5
    samples = 1000
    for _ in range(samples):
        profile = uniform_profile(n, m)
        winners = stv.elect(profile, k)
        assert len(winners) == k


def test_fractional_vs_weighted_transfer(fractional_vs_weighted_transfer_profile):
    frac_stv = STV(transfer_type = 'fractional')
    weighted_frac_stv = STV(transfer_type = 'weighted-fractional')
    
    k = 3
    frac_winners = frac_stv.elect(fractional_vs_weighted_transfer_profile, k)
    weighted_frac_winners = weighted_frac_stv.elect(fractional_vs_weighted_transfer_profile, k)
     
    assert set(frac_winners) == {0,2,3}
    assert set(weighted_frac_winners) == {0,1,2}
    

def test_weighted_fractional_with_votekit():
    n = 20
    m = 8
    k = 5
    #np.random.seed(99877)
    prof = uniform_profile(n, m)
    ballots = [Ballot(ranking = [{str(j)} for j in prof[:,i]], weight = 1) for i in range(n)]
    votekit_prof = PreferenceProfile(ballots= ballots)
    
    samples = 1000
    winner_dist = {frozenset(comb): 0 for comb in itertools.combinations(range(m), k)}
    votekit_winner_dist = {frozenset(comb): 0 for comb in itertools.combinations(range(m), k)}
    for i in range(samples):
        stv = STV(transfer_type = 'weighted-fractional')
        winners = stv.elect(prof, k)
        winner_dist[frozenset(winners)] += 1
        
        with contextlib.redirect_stdout(None):
            election = votekit_STV(votekit_prof, k, transfer = fractional_transfer)
            votekit_winners = election.get_elected()
            
        votekit_winners = [int(winner) for winner_set in
                            votekit_winners for winner in list(winner_set)]
        votekit_winner_dist[frozenset(votekit_winners)] += 1
        
    tv_distance = 0
    for wset in winner_dist.keys():
        tv_distance += abs(winner_dist[wset]/samples - votekit_winner_dist[wset]/samples)
    tv_distance /= 2
    
    #breakpoint()
    assert tv_distance < 0.05


def test_cambridge_with_votekit():
    n = 20
    m = 8
    k = 5
    #np.random.seed(99877)
    prof = uniform_profile(n, m)
    ballots = [Ballot(ranking = [{str(j)} for j in prof[:,i]], weight = 1) for i in range(n)]
    votekit_prof = PreferenceProfile(ballots= ballots)
    
    samples = 1000
    winner_dist = {frozenset(comb): 0 for comb in itertools.combinations(range(m), k)}
    votekit_winner_dist = {frozenset(comb): 0 for comb in itertools.combinations(range(m), k)}
    for i in range(samples):
        stv = STV(transfer_type = 'cambridge')
        winners = stv.elect(prof, k)
        winner_dist[frozenset(winners)] += 1
        
        with contextlib.redirect_stdout(None):
            election = votekit_STV(votekit_prof, k, transfer = random_transfer)
            votekit_winners = election.get_elected()
            
        votekit_winners = [int(winner) for winner_set in
                            votekit_winners for winner in list(winner_set)]
        votekit_winner_dist[frozenset(votekit_winners)] += 1
        
    tv_distance = 0
    for wset in winner_dist.keys():
        tv_distance += abs(winner_dist[wset]/samples - votekit_winner_dist[wset]/samples)
    tv_distance /= 2
    
    #breakpoint()
    assert tv_distance < 0.05