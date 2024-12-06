import pytest
import numpy as np

@pytest.fixture
def incomplete_profile():
   return np.array([
       [1, 3, 1, 3, 3, 0, 2, 3, 2, 1],
       [0, 0, 0, 0, 0, 1, 3, 0, 0, 0],
       [3, 2, 3, 2, 1, 3, 0, 2, 3, 3],
    ])
    

@pytest.fixture
def profile_with_repeats():
   return np.array([
       [1, 3, 1, 3, 3, 0, 2, 3, 2, 1],
       [0, 0, 0, 0, 0, 1, 3, 0, 0, 0],
       [0, 2, 3, 2, 1, 3, 0, 2, 3, 1],
       [2, 1, 2, 1, 2, 2, 1, 1, 1, 2]
    ])

@pytest.fixture
def basic_profile():
   return np.array([
       [1, 3, 1, 3, 3, 0, 2, 3, 2, 1],
       [0, 0, 0, 0, 0, 1, 3, 0, 0, 0],
       [3, 2, 3, 2, 1, 3, 0, 1, 3, 3],
       [2, 1, 2, 1, 2, 2, 1, 2, 1, 2]
    ])

@pytest.fixture
def profile_with_fp_tie():
   return np.array([
       [1, 3, 1, 3, 3, 0, 1, 3, 2, 1],
       [0, 0, 0, 0, 0, 1, 3, 0, 0, 0],
       [3, 2, 3, 2, 1, 3, 0, 2, 3, 3],
       [2, 1, 2, 1, 2, 2, 2, 1, 1, 2]
    ])
    
@pytest.fixture
def profile_with_bloc_tie():
   return np.array([
       [1, 3, 1, 3, 3, 0, 2, 3, 2, 1],
       [0, 0, 0, 0, 0, 1, 3, 0, 0, 0],
       [3, 2, 3, 2, 1, 3, 0, 2, 3, 3],
       [2, 1, 2, 1, 2, 2, 1, 1, 1, 2]
    ])
   
@pytest.fixture
def profile_with_fp_borda_tie():
   return np.array([
       [1, 3, 1, 3, 3, 0, 2, 3, 2, 1],
       [0, 0, 0, 0, 0, 3, 3, 0, 0, 0],
       [3, 2, 3, 2, 1, 1, 0, 2, 3, 3],
       [2, 1, 2, 1, 2, 2, 1, 1, 1, 2]
    ])  
   
@pytest.fixture
def profile_with_full_borda_tie():
   return np.array([
       [0, 1, 2, 3, 0, 1, 2, 3],
       [1, 2, 3, 0, 1, 2, 3, 0],
       [2, 3, 0, 1, 2, 3, 0, 1],
       [3, 0, 1, 2, 3, 0, 1, 2]
      ])  
   
   
@pytest.fixture
def fractional_vs_weighted_transfer_profile():
   return np.array([
      [0,0,0,0,0,0,0,1,2,2],
      [2,2,2,2,2,2,1,0,0,0],
      [3,3,3,3,3,3,2,2,1,3],
      [1,1,1,1,1,1,3,3,3,1]
   ])
   
   
@pytest.fixture
def basic_profile_with_elimination_tie():
   return np.array([
      [0,0,0,0,1,1,1,2,2,2],
      [1,1,1,1,0,0,0,0,0,0],
      [2,2,2,2,2,2,2,1,1,1]
   ])
   

@pytest.fixture
def basic_profile_for_fractional_transfer():
   return np.array([
      [0,0,0,0,0,0,1,1,1,2],
      [1,1,1,2,3,3,0,0,0,1],
      [2,2,2,3,2,2,2,2,2,0],
      [3,3,3,1,1,1,3,3,3,3]
   ])
   
   
@pytest.fixture
def complex_stv_profile():
   return np.array([
      [0, 7, 5, 1, 0, 4, 1, 0, 1, 2, 0, 7, 5, 2, 5, 3, 4, 4, 5, 7],
      [5, 2, 0, 0, 6, 1, 0, 1, 5, 6, 3, 4, 7, 3, 7, 2, 5, 6, 0, 3],
      [4, 3, 4, 7, 7, 0, 4, 7, 7, 5, 2, 6, 3, 0, 0, 1, 1, 3, 2, 5],
      [6, 0, 2, 6, 4, 7, 5, 6, 4, 7, 1, 0, 2, 1, 1, 5, 0, 1, 7, 0],
      [1, 5, 3, 2, 5, 6, 6, 3, 2, 1, 7, 1, 6, 5, 6, 6, 7, 0, 4, 2],
      [3, 1, 1, 4, 2, 3, 3, 4, 3, 0, 5, 2, 0, 7, 3, 7, 6, 7, 3, 1],
      [7, 4, 7, 5, 1, 2, 2, 5, 6, 4, 4, 5, 1, 6, 2, 0, 2, 2, 6, 4],
      [2, 6, 6, 3, 3, 5, 7, 2, 0, 3, 6, 3, 4, 4, 4, 4, 3, 5, 1, 6]
   ])
   
   
@pytest.fixture
def basic_monroe_profile():
   return np.array([
       [1, 3, 1, 3, 3, 0, 2, 3, 2, 1],
       [3, 0, 0, 0, 0, 1, 3, 0, 3, 0],
       [0, 2, 3, 2, 1, 3, 0, 1, 0, 3],
       [2, 1, 2, 1, 2, 2, 1, 2, 1, 2]
    ])
   

@pytest.fixture
def monroe_vs_chamberlin_profile():
   return np.array([
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [3, 3, 3, 3, 3, 2, 2, 2, 2, 2],
       [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])


@pytest.fixture
def plurality_veto_elim_profile():
   return np.array([
      [1,2,3,1,3,3],
      [0,0,0,0,0,0],
      [3,1,2,3,2,1],
      [2,3,1,2,1,2]
   ])
   
@pytest.fixture
def plurality_veto_elim_tie_profile():
   return np.array([
      [0,0,0,0,0,0],
      [1,1,1,1,1,1],
      [2,2,2,2,2,2],
      [3,3,3,3,3,3]
   ])
   
@pytest.fixture
def expanding_fp_tie_profile():
   return np.array([
      [1,0,1,0,1,0,1,0],
      [0,1,0,1,0,1,0,1],
      [2,2,2,2,2,2,2,2],
      [3,3,3,3,3,3,3,3]
   ])
   