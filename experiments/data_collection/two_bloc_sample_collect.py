import numpy as np
from metric_voting.spatial_generation import *
from metric_voting.election_sampling import samples
from metric_voting.elections import *


# Choose number of voters n
# And the number of candidates m
n = 1000
m = 20

# And the number of winners for the election
k = 4 

# Means for each of the 2 Gaussian distributions
means = [[0, -1], [0, 1]]
stds = [1/6, 1/6]  # Standard deviations for each Gaussian
group_sizes = [500,500]  # Group Sizes

voter_params = [{'loc': None, 'scale': None, 'size': 2} for _ in range(len(group_sizes))]
for i,mean in enumerate(means):
    voter_params[i]['loc'] = mean

for i,std in enumerate(stds):
    voter_params[i]['scale'] = std
    
candidate_params = [{'low': -2, 'high': 2, 'size': 2}]

distance = lambda point1, point2: np.linalg.norm(point1 - point2)

two_party_generator = GroupSpatial(
    n_voter_groups = 2,
    n_candidate_groups = 1, 
    voter_dist_fns = [np.random.normal]*len(group_sizes),
    voter_dist_fn_params = voter_params,
    candidate_dist_fns = [np.random.uniform],
    candidate_dist_fn_params = candidate_params,
    distance_fn = distance
)

# Define elections
elections_dict = {SNTV:{}, Bloc:{}, Borda:{}, Harmonic:{'solver' : 'GUROBI_CMD'},
                  STV:{'transfer_type' : 'weighted-fractional'},
                 ChamberlinCourant:{'solver' : 'GUROBI_CMD'}, GreedyCC:{},
                  Monroe:{'solver' : 'GUROBI_CMD'}, GreedyMonroe:{}, 
                  PluralityVeto:{}, CommitteeVeto:{'q':k}, 
                  ExpandingApprovals: {},
                 SMRD:{}, OMRD:{}, DMRD:{'rho': 0.5}}

# Number of samples to use
n_samples = 10000

# set the seed for deterministic results:
np.random.seed(918717)

# and sample from them
f = 'data/2bloc.npz'

generator_input = [
    {'voter_group_sizes': group_sizes,
     'candidate_group_sizes': [m]}
]

result_list = samples(
    n_samples,
    two_party_generator,
    elections_dict,
    generator_input,
    k,
    dim = 2,
    filename = f
)