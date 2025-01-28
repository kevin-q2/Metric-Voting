import numpy as np
from metric_voting.measurements import *


def test_cost_array():
    voter_positions = np.zeros((4, 2))
    candidate_positions = np.array([
        [3, 4],
        [6, 8],
        [5, 12],
        [9, 12]
    ])
    
    cst_array = cost_array(voter_positions, candidate_positions)
    assert np.all(cst_array == np.array([
        [5, 5, 5, 5],
        [10, 10, 10, 10],
        [13, 13, 13, 13],
        [15, 15, 15, 15]
    ]))
    
    
    
def test_euclidean_cost_array():
    voter_positions = np.zeros((4, 2))
    candidate_positions = np.array([
        [3, 4],
        [6, 8],
        [5, 12],
        [9, 12]
    ])
    
    cst_array = euclidean_cost_array(voter_positions, candidate_positions)
    assert np.all(cst_array == np.array([
        [5, 5, 5, 5],
        [10, 10, 10, 10],
        [13, 13, 13, 13],
        [15, 15, 15, 15]
    ]))
    
    
    
def test_random_cost_array():
    samples = 100
    for _ in range(samples):
        voter_positions = np.random.normal(size = (100, 10))
        candidate_positions = np.random.uniform(size = (43, 10))

        cst_array1 = cost_array(voter_positions, candidate_positions)
        cst_array2 = euclidean_cost_array(voter_positions, candidate_positions)
        
        assert cst_array1.shape == (43,100)
        assert cst_array2.shape == (43,100)     
        assert np.allclose(cst_array1, cst_array2)
        
        
def test_min_assignment():
    cst_array = np.array([
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [9,10,11,12],
        [13,14,15,16]   
    ])
    
    assert set(min_assignment(cst_array, 1)) == {0}
    assert set(min_assignment(cst_array, 2)) == {0,1}
    assert set(min_assignment(cst_array, 4)) == {0,1,2,3}
    assert set(min_assignment(cst_array, 5)) == {0,1,2,3,4}
    
    # Randomly break ties for third place:
    samples = 1000
    third_min = np.zeros(samples)
    for i in range(samples):
        third_place = min_assignment(cst_array, 3)[2]
        third_min[i] = third_place
        
    _, counts = np.unique(third_min, return_counts=True)
    assert  len(counts) == 2
    assert np.allclose(counts[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts[1]/samples, 0.5, atol = 0.05, rtol = 0)
    
    
def test_min_assignment_cost():
    cst_array = np.array([
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [9,10,11,12],
        [13,14,15,16]   
    ])
    
    assert min_assignment_cost(cst_array, 1) == 10
    assert min_assignment_cost(cst_array, 2) == 36
    assert min_assignment_cost(cst_array, 3) == 78
    assert min_assignment_cost(cst_array, 4) == 120
    assert min_assignment_cost(cst_array, 5) == 178
    
    

def test_q_costs():
    cst_array = np.array([
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [9,10,11,12],
        [13,14,15,16]   
    ])
    
    q_cst = q_costs(1, cst_array)
    assert set(q_cst) == {1,2,3,4}
    
    q_cst = q_costs(2, cst_array)
    assert set(q_cst) == {5,6,7,8}
    
    q_cst = q_costs(3, cst_array)
    assert set(q_cst) == {9,10,11,12}
    
    q_cst = q_costs(4, cst_array)
    assert set(q_cst) == {9,10,11,12}
    
    q_cst = q_costs(5, cst_array)
    assert set(q_cst) == {13,14,15,16}


def test_q_cost_array():
    cst_array = np.array([
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [9,10,11,12],
        [13,14,15,16]   
    ])
    
    q_cst = q_cost_array(1, cst_array, [{0,1,2,3,4}])
    assert np.array_equal(q_cst, np.array([[1,2,3,4]]))
    
    q_cst = q_cost_array(2, cst_array, [{0,3}, {1,4}])
    assert np.array_equal(q_cst, np.array([[9,10,11,12],[13,14,15,16]]))