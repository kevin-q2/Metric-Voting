import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import seaborn as sns
import itertools as it
import pulp
from sklearn.cluster import KMeans
import time

sys.path.append(os.path.join(os.getcwd(), 'metric_voting/code'))
from spatial_generation import Spatial, GroupSpatial
from elections import SNTV,Bloc,STV,Borda, ChamberlinCourant, Monroe, GreedyCC, PluralityVeto, SMRD, OMRD, DMRD
from tools import cost, best_group_cost, worst_group_cost, representativeness, representativeness_ratio, remove_candidates, borda_matrix, group_representation, max_group_representation
from election_sampling import election_sample, samples


# Colors!
pal = sns.color_palette("hls", 8)
tab20_colors = plt.cm.tab20.colors


# Choose number of voters n
# And the number of candidates m
n = 10000
m = 20

# And the number of winners for the election
k = 4

# Means for each of the 4 Gaussian distributions
means = [[-2, 0], [2, 0], [0, 2], [0, -2]]
stds = [0.5, 0.5, 0.5, 0.5]  # Standard deviations for each Gaussian
four_party_G = [2500, 2500, 2500, 2500]  # Group Sizes

voter_params = [{'loc': None, 'scale': None, 'size': 2} for _ in range(len(four_party_G))]
for i,mean in enumerate(means):
    voter_params[i]['loc'] = mean

for i,std in enumerate(stds):
    voter_params[i]['scale'] = std
    
candidate_params = {'low': -3, 'high': 3, 'size': 2}

distance = lambda point1, point2: np.linalg.norm(point1 - point2)

four_party_generator = GroupSpatial(m = m, g = len(four_party_G),
                    voter_dists = [np.random.normal]*len(four_party_G), voter_params = voter_params,
                    candidate_dist = np.random.uniform, candidate_params = candidate_params,
                    distance = distance)


# Generate a profile from random candidate and voter positions
profile, candidate_positions, voter_positions, voter_labels = four_party_generator.generate(four_party_G)


# Define elections
elections_dict = {SNTV:{}, Bloc:{}, STV:{},
                 Borda:{}, GreedyCC:{}, PluralityVeto:{},
                 SMRD:{}, OMRD:{}, DMRD:{'rho': 0.5}}
elections_list = [SNTV, Bloc, STV, Borda, GreedyCC, 
                  PluralityVeto, SMRD, OMRD, DMRD]
n_samples = 10000

# and sample from them
f = 'metric_voting/data/4party_10k.npz'
results_list = samples(n_samples, four_party_generator, elections_dict, [four_party_G], k, dim = 2, filename = f)
result_dict = results_list[0]

f = 'metric_voting/data/4party_10k_1.npz'
results_list = samples(n_samples, four_party_generator, elections_dict, [four_party_G], k = 5, dim = 2, filename = f)
result_dict = results_list[0]