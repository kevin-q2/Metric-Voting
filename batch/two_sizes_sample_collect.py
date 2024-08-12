import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import itertools as it
import pulp
from sklearn.cluster import KMeans
import time
pal = sns.color_palette("hls", 8)

sys.path.append(os.path.join(os.getcwd(), 'metric_voting/code'))
from spatial_generation import Spatial, GroupSpatial
from elections import SNTV,STV,Borda,RandomDictator,PRD, PluralityVeto
from tools import cost, best_group_cost, representativeness, remove_candidates
from election_sampling import election_sample, samples
from spatial_generation import Spatial, GroupSpatial
from elections import SNTV,Bloc,STV,Borda,RandomDictator,PRD, PluralityVeto, ChamberlinCourant, Monroe, GreedyCC, RandomDictator2
from tools import cost, best_group_cost, worst_group_cost, representativeness, representativeness_ratio, remove_candidates, borda_matrix
from election_sampling import election_sample, samples



# Choose number of voters n
# And the number of candidates m
n = 100
m = 20

# And the number of winners for the election
k = 4 

# Means for each of the 2 Gaussian distributions
means = [[0, -1.5], [0, 1.5]]
stds = [0.5, 0.5]  # Standard deviations for each Gaussian
two_party_G = [50,50]  # Group Sizes

voter_params = [{'loc': None, 'scale': None, 'size': 2} for _ in range(len(two_party_G))]
for i,mean in enumerate(means):
    voter_params[i]['loc'] = mean

for i,std in enumerate(stds):
    voter_params[i]['scale'] = std
    
candidate_params = {'low': -3, 'high': 3, 'size': 2}

distance = lambda point1, point2: np.linalg.norm(point1 - point2)

two_party_generator = GroupSpatial(m = m, g = len(two_party_G),
                    voter_dists = [np.random.normal]*len(two_party_G), voter_params = voter_params,
                    candidate_dist = np.random.uniform, candidate_params = candidate_params,
                    distance = distance)


# Generate a profile from random candidate and voter positions
profile, candidate_positions, voter_positions, voter_labels = two_party_generator.generate(two_party_G)

group_sizes = [[i, 100 - i] for i in range(50, -1, -5)]


# Define elections
elections_dict = {SNTV:{}, Bloc:{}, STV:{},
                 Borda:{}, ChamberlinCourant:{}, GreedyCC:{}, Monroe:{},
                  RandomDictator:{'rho': 0.5}, RandomDictator2:{}, PRD:{'rho': 0.5}, PluralityVeto:{}}
elections_list = [SNTV, Bloc, STV, Borda, ChamberlinCourant, GreedyCC, Monroe, 
                  RandomDictator,RandomDictator2, PRD, PluralityVeto]
n_samples = 1000

# and sample from them
f = 'metric_voting/data/2sizes.npz'
result_dict = samples(n_samples, two_party_generator, elections_dict, group_sizes, k, dim = 2, filename = f)




