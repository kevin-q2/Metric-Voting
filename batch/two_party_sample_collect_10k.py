import sys
import os
import numpy as np
import pandas as pd
import itertools as it
import pulp
from sklearn.cluster import KMeans
import time

sys.path.append(os.path.join(os.getcwd(), 'metric_voting/code'))
from spatial_generation import Spatial, GroupSpatial
from elections import SNTV,Bloc,STV,Borda, ChamberlinCourant, Monroe, GreedyCC, PluralityVeto,ExpandingApprovals, SMRD, OMRD, DMRD
from election_sampling import election_sample, samples


# Choose number of voters n
# And the number of candidates m
n = 10000
m = 20

# And the number of winners for the election
k = 4 

# Means for each of the 2 Gaussian distributions
means = [[0, -2], [0, 2]]
stds = [0.5, 0.5]  # Standard deviations for each Gaussian
two_party_G = [5000,5000]  # Group Sizes

voter_params = [{'loc': None, 'scale': None, 'size': 2} for _ in range(len(two_party_G))]
for i,mean in enumerate(means):
    voter_params[i]['loc'] = mean

for i,std in enumerate(stds):
    voter_params[i]['scale'] = std
    
candidate_params = {'low': -5, 'high': 5, 'size': 2}

distance = lambda point1, point2: np.linalg.norm(point1 - point2)

two_party_generator = GroupSpatial(m = m, g = len(two_party_G),
                    voter_dists = [np.random.normal]*len(two_party_G), voter_params = voter_params,
                    candidate_dist = np.random.uniform, candidate_params = candidate_params,
                    distance = distance)


# Generate a profile from random candidate and voter positions
profile, candidate_positions, voter_positions, voter_labels = two_party_generator.generate(two_party_G)


# Define elections
elections_dict = {SNTV:{}, Bloc:{}, STV:{},
                 Borda:{}, GreedyCC:{}, PluralityVeto:{}, ExpandingApprovals:{},
                 SMRD:{}, OMRD:{}, DMRD:{'rho': 0.5}}
n_samples = 10000

# set the seed for deterministic results:
np.random.seed(918717)

# and sample from them
f = 'metric_voting/data/2party1_10k.npz'
results_list = samples(n_samples, two_party_generator, elections_dict, [two_party_G], k, dim = 2, filename = f)
result_dict = results_list[0]