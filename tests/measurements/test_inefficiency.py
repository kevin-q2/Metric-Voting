import numpy as np
from itertools import combinations
from metric_voting.measurements import *


def test_proportional_assignment():
    cst_array = np.array([
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [9,10,11,12],
        [13,14,15,16]
    ])
    voter_labels = np.array([1,0,0,1])
    assignment = proportional_assignment(cst_array, voter_labels, bloc_label = 1, k = 2)
    assert len(assignment) == 1
    assert set(assignment) == {0}
    
    voter_labels = np.array([1,1,0,1])
    assignment = proportional_assignment(cst_array, voter_labels, bloc_label = 1, k = 2)
    assert len(assignment) == 1
    assert set(assignment) == {0}
    
    voter_labels = np.array([1,1,1,1])
    assignment = proportional_assignment(cst_array, voter_labels, bloc_label = 1, k = 2)
    assert len(assignment) == 2
    assert set(assignment) == {0,1}
    
    
def test_proportional_assignment_cost():
    cst_array = np.array([
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [9,10,11,12],
        [13,14,15,16]
    ])
    voter_labels = np.array([1,0,0,1])
    assignment_cost = proportional_assignment_cost(cst_array, voter_labels, bloc_label = 1, k = 2)
    assert assignment_cost == 5
    
    voter_labels = np.array([1,1,0,1])
    assignment_cost = proportional_assignment_cost(cst_array, voter_labels, bloc_label = 1, k = 2)
    assert assignment_cost == 7
    
    voter_labels = np.array([1,1,1,1])
    assignment_cost = proportional_assignment_cost(cst_array, voter_labels, bloc_label = 1, k = 2)
    assert assignment_cost == 36
    
    
def test_group_inefficiency():
    candidate_positions = np.array(
        [[-1,1]] * 4 +
        [[0,1]] * 4 +
        [[1,1]] * 4 +
        [[-1,0]] * 4 +
        [[0,0]] * 4 +
        [[1,0]] * 4 +
        [[-1,-1]] * 4 +
        [[0,-1]] * 4 +
        [[1,-1]] * 4
    )

    n = 10000
    bloc1 = np.zeros((n,2)) + np.array([0.1,1])
    bloc2 = np.zeros((n,2)) + np.array([-0.1,-1])
    d1 = 0.1
    d2 = np.linalg.norm(np.array([0.1,1]) - np.array([0,-1]))
    d3 = np.linalg.norm(np.array([0.1,1]) - np.array([0, 0]))
    voter_positions = np.vstack((bloc1, bloc2))
    cst_array = euclidean_cost_array(voter_positions, candidate_positions)
    
    group_labels = np.zeros(2*n)
    group_labels[:n] = 1
    overall_labels = np.ones(2*n)
    
    # 2-2 split:
    winner_indices = [4,6,29,30]
    group_ineff = group_inefficiency(cst_array, winner_indices, group_labels, bloc_label = 1)
    assert group_ineff == 1.0
        
    overall_ineff = group_inefficiency(cst_array, winner_indices, overall_labels, bloc_label = 1)
    est = (n*2*d1 + n*2*d2) / (n*4*d3)
    assert np.isclose(overall_ineff, est)
    
    # 3-1 split:
    winner_indices = [4,28,29,30]
    group_ineff = group_inefficiency(cst_array, winner_indices, group_labels, bloc_label = 1)
    est = (n* d1 + n * d2)/(n * 2 * d1)
    assert np.isclose(group_ineff, est)
    
    overall_ineff = group_inefficiency(cst_array, winner_indices, overall_labels, bloc_label = 1)
    est = (n * d1 + n * 3 * d2 + n * 3 * d1 + n * d2) / (2 * n * 4 * d3)
    assert np.isclose(overall_ineff, est)
    
    # 4-0 split:
    winner_indices = [28,29,30,31]
    group_ineff = group_inefficiency(cst_array, winner_indices, group_labels, bloc_label = 1)
    est = d2/d1
    assert np.isclose(group_ineff, est)
    
    overall_ineff = group_inefficiency(cst_array, winner_indices, overall_labels, bloc_label = 1)
    est = (n * 4 * d2 + n * 4 * d1) / (2 * n * 4 * d3)
    assert np.isclose(overall_ineff, est)
    
    # All middle:
    winner_indices = [16,17,18,19]
    group_ineff = group_inefficiency(cst_array, winner_indices, group_labels, bloc_label = 1)
    est = (n * 2 * d3)/(n * 2 * d1)
    assert np.isclose(group_ineff, est)
    
    overall_ineff = group_inefficiency(cst_array, winner_indices, overall_labels, bloc_label = 1)
    assert overall_ineff == 1.0
    
    
    
def test_with_random_cost_array():
    samples = 100
    for _ in range(samples):
        voter_positions = np.random.normal(size = (100, 10))
        candidate_positions = np.random.uniform(size = (43, 10))
        cst_array = euclidean_cost_array(voter_positions, candidate_positions)
        
        winners = np.random.choice(43, 10, replace = False)
        overall_labels = np.ones(100)
        overall_ineff = group_inefficiency(cst_array, winners, overall_labels, bloc_label = 1)
        assert overall_ineff >= 1.0
        

def test_random_group_ineff():
    cst_array = np.array([
        [0,2,0,4],
        [5,6,7,8],
        [0,10,0,12],
        [9,10,11,12],
        [13,14,15,16]
    ])
    winner_indices = [0,2]
    est = group_inefficiency(cst_array, winner_indices, np.array([0,1,0,1]), bloc_label = 1)
    rineff, rbloc = random_group_inefficiency(cst_array, winner_indices, t = 1)
    assert rineff == est
    
    cst_array = np.array([
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [9,10,11,12],
        [13,14,15,16]
    ])
    winner_indices = [3,4]
    
    ineffs = []
    for i, comb in enumerate(combinations(range(4), 2)):
        labels = np.zeros(4)
        labels[list(comb)] = 1
        ineffs.append(group_inefficiency(cst_array, winner_indices, labels, bloc_label = 1))
        
    for i, comb in enumerate(combinations(range(4), 3)):
        labels = np.zeros(4)
        labels[list(comb)] = 1
        ineffs.append(group_inefficiency(cst_array, winner_indices, labels, bloc_label = 1))
        
    samples = 10000
    random_ineffs = np.zeros(samples)
    for i in range(samples): 
        rineff,rbloc = random_group_inefficiency(
            cst_array,
            winner_indices,
            t = 1,
            weights = np.ones(4)/4
        )
        random_ineffs[i] = rineff
        
    unique1,counts1 = np.unique(ineffs, return_counts=True)
    unique2,counts2 = np.unique(random_ineffs, return_counts=True)
    assert np.allclose(unique1, unique2, atol = 0.05)
    for i,c in enumerate(counts1):
        assert np.isclose(c/len(ineffs), counts2[i]/samples, atol = 0.05, rtol = 0.05)
        
    
        
def test_heuristic_bloc():
    candidate_positions = np.array(
        [[0,4]] * 4 +
        [[0,8]] * 4 + 
        [[0,12]] * 4
    )

    n = 10000
    bloc1 = np.zeros((n,2)) + np.array([3,4])
    bloc2 = np.zeros((n,2)) + np.array([3,12])
    voter_positions = np.vstack((bloc1, bloc2))
    cst_array = euclidean_cost_array(voter_positions, candidate_positions)
    
    # 4-0 split:
    winner_indices = [8,9,10,11]
    bloc = heuristic_worst_bloc(cst_array, winner_indices)
    assert set(bloc) == set(range(n))
    
    # Middle split:
    winner_indices = [4,5,6,7]
    samples = 1000
    blocs = np.zeros((samples, n))
    for i in range(samples):
        bloc = heuristic_worst_bloc(cst_array, winner_indices)
        assert len(bloc) == n
        blocs[i] = bloc
        
    unique,counts = np.unique(blocs, return_counts=True)
    assert len(unique) == 2*n
    for c in counts:
        assert np.isclose(c/samples, 0.5, atol = 0.05, rtol = 0.05)