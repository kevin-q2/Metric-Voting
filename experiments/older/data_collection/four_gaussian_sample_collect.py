import sys
import os
import numpy as np
import pandas as pd
import itertools as it
import pulp
import time

#sys.path.append(os.path.join(os.getcwd(), 'metric_voting/code'))
sys.path.append('main')
from spatial_generation import Spatial, GroupSpatial
from elections import SNTV,Bloc,STV,Borda, ChamberlinCourant, Monroe, GreedyCC, PluralityVeto, SMRD, OMRD, DMRD, ExpandingApprovals
from election_sampling import election_sample, samples


# Choose number of voters n
# And the number of candidates m
n = 200
m = 200

# And the number of winners for the election
k = 20

# Means for each of the 4 Normal distributions
means = [[-2, 0], [2, 0], [0, 2], [0, -2]]
stds = [0.5, 0.5, 0.5, 0.5]  # Standard deviations for each Normal
four_party_G = [25, 25, 25, 25]  # Group Sizes

# Create a list of voter parameters -- with each set of parameters being a dict
voter_params = [{'loc': None, 'scale': None, 'size': 2} for _ in range(len(four_party_G))]
for i,mean in enumerate(means):
    voter_params[i]['loc'] = mean

for i,std in enumerate(stds):
    voter_params[i]['scale'] = std

# define a distance function between voters and candidates
distance = lambda point1, point2: np.linalg.norm(point1 - point2)

# Create the group spatial generator object!
four_party_generator = GroupSpatial(m = m, g = len(four_party_G),
                    voter_dists = [np.random.normal]*len(four_party_G), voter_params = voter_params,
                    candidate_dists = [np.random.normal]*len(four_party_G), candidate_params = voter_params,
                    distance = distance)

# Define elections
elections_dict = {SNTV:{}, Bloc:{}, STV:{},
                 Borda:{}, ChamberlinCourant:{}, GreedyCC:{}, Monroe:{}, PluralityVeto:{},
                 #Borda:{}, GreedyCC:{}, PluralityVeto:{},
                 ExpandingApprovals: {}, SMRD:{}, OMRD:{}, DMRD:{'rho': 0.5}}

# Number of samples to use
n_samples = 1

# set the seed for deterministic results:
np.random.seed(918717)

# and sample from them
f = 'data/four_gaussian_full.npz'
size_dist = [1/4, 1/4, 1/4, 1/4]

import time
start = time.time()
results_list = samples(n_samples, four_party_generator, elections_dict, 
                       [[n, size_dist, size_dist, False]], k, dim = 2, filename = f)
end = time.time()
print("Four gaussian time taken: " + str(end - start))


result_dict = results_list[0]