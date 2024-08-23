import sys
import os
import numpy as np
import pandas as pd
import itertools as it
import pulp
import time

#sys.path.append(os.path.join(os.getcwd(), 'metric_voting/code'))
sys.path.append('../code')
from spatial_generation import Spatial, GroupSpatial
from elections import SNTV,Bloc,STV,Borda, ChamberlinCourant, Monroe, GreedyCC, PluralityVeto, SMRD, OMRD, DMRD, ExpandingApprovals
from election_sampling import election_sample, samples


# Choose number of voters n
# And the number of candidates m
n = 100
m = 20

# And the number of winners for the election
k = 5

# Means for each of the 4 Gaussian distributions
means = [[-2, 0], [2, 0], [0, 2], [0, -2]]
stds = [0.5, 0.5, 0.5, 0.5]  # Standard deviations for each Gaussian
four_party_G = [25, 25, 25, 25]  # Group Sizes

voter_params = [{'loc': None, 'scale': None, 'size': 2} for _ in range(len(four_party_G))]
for i,mean in enumerate(means):
    voter_params[i]['loc'] = mean

for i,std in enumerate(stds):
    voter_params[i]['scale'] = std
    
candidate_params = {'low': -5, 'high': 5, 'size': 2}

distance = lambda point1, point2: np.linalg.norm(point1 - point2)

four_party_generator = GroupSpatial(m = m, g = len(four_party_G),
                    voter_dists = [np.random.normal]*len(four_party_G), voter_params = voter_params,
                    candidate_dist = np.random.uniform, candidate_params = candidate_params,
                    distance = distance)


# Define elections
elections_dict = {SNTV:{}, Bloc:{}, STV:{},
                 Borda:{}, ChamberlinCourant:{}, GreedyCC:{}, Monroe:{}, PluralityVeto:{},
                 ExpandingApprovals: {}, SMRD:{}, OMRD:{}, DMRD:{'rho': 0.5}}

# Number of samples to use
n_samples = 10000

# set the seed for deterministic results:
np.random.seed(918717)

# and sample from them
f = 'metric_voting/data/4party1_5cand.npz'
results_list = samples(n_samples, four_party_generator, elections_dict, [four_party_G], k, dim = 2, filename = f)
result_dict = results_list[0]