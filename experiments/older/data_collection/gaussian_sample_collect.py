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

# define the single set of voter and candidate paramters
voter_params = {'loc': [0,0], 'scale': 1, 'size': 2}
candidate_params = {'loc': [0,0], 'scale': 1, 'size': 2}

# define a distance function between voters and candidates
distance = lambda point1, point2: np.linalg.norm(point1 - point2)

# Create the group spatial generator object!
gaussian_generator = Spatial(m = m, voter_dist = np.random.normal, voter_params = voter_params,
                    candidate_dist = np.random.normal, candidate_params = candidate_params,
                    distance = distance)


# Define elections
elections_dict = {SNTV:{}, Bloc:{}, STV:{},
                 Borda:{}, ChamberlinCourant:{}, GreedyCC:{}, Monroe:{}, PluralityVeto:{},
                 #Borda:{}, GreedyCC:{}, PluralityVeto:{},
                 ExpandingApprovals: {}, SMRD:{}, OMRD:{}, DMRD:{'rho': 0.5}}

# Number of samples to use
n_samples = 10000

# set the seed for deterministic results:
np.random.seed(918717)

# and sample from them
f = 'data/gaussian_full.npz'
results_list = samples(n_samples, gaussian_generator, elections_dict, [n], k, dim = 2, filename = f)
result_dict = results_list[0]