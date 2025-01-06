import numpy as np
from metric_voting import *


scot_profile = np.load('../../data/scot-elex/aberdeen_2022_ward11.npy')

m = scot_profile.shape[0]
n = scot_profile.shape[1]
k = 3
d = 7
#candidate_positions = np.random.uniform(0,1, size = (m,d))
candidate_positions = np.zeros((m,d))
np.fill_diagonal(candidate_positions,1)
voter_dist_fn = np.random.uniform
voter_dist_fn_params = {'low' : 0, 'high' : 2, 'size' : d}
distance = lambda point1, point2: np.linalg.norm(point1 - point2)

ranked_generator = RankedSpatial(
    voter_dist_fn,
    voter_dist_fn_params
)

# Define elections
elections_dict = {SNTV:{}, Bloc:{}, Borda:{},
                  STV:{'transfer_type' : 'weighted-fractional'},
                 ChamberlinCourant:{'solver' : 'GUROBI_CMD'}, GreedyCC:{},
                  Monroe:{'solver' : 'GUROBI_CMD'}, 
                  PluralityVeto:{}, CommitteeVeto:{'q':k}, 
                  ExpandingApprovals: {},
                 SMRD:{}, OMRD:{}, DMRD:{'rho': 0.5}}

#elections_dict = {Harmonic:{'solver' : 'GUROBI_CMD'}}


# Number of samples to use
n_samples = 100

# set the seed for deterministic results:
np.random.seed(918717)

# and sample from them
f = 'data/aberdeen_2022_ward11_results.npz'

generator_input = [
    {'profile': scot_profile,
     'candidate_positions': candidate_positions}
]

result_list = samples(
    n_samples,
    ranked_generator,
    elections_dict,
    generator_input,
    k,
    dim = 2,
    filename = f
)