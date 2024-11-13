import sys
import os
import numpy as np
import pandas as pd
import itertools as it
import pulp
import time

#sys.path.append(os.path.join(os.getcwd(), 'metric_voting/code'))
sys.path.append('code')
from spatial_generation import Spatial, GroupSpatial
from elections import SNTV,Bloc,STV,Borda, ChamberlinCourant, Monroe, GreedyCC, PluralityVeto, SMRD, OMRD, DMRD, ExpandingApprovals
from election_sampling import election_sample, samples


# Choose number of voters n
# And the number of candidates m
n = 200
m = 200

# And the number of winners for the election
k = 20

# samples a single point uniformly from a disc with defined radius
def sample_uniform_disc(radius=1):
    # Sample angles uniformly from 0 to 2*pi
    theta = np.random.uniform(0, 2 * np.pi, 1)
    
    # Sample radii with correct distribution
    r = radius * np.sqrt(np.random.uniform(0, 1, 1))
    
    # Convert polar coordinates to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))[0]

voter_params = {'radius': 1}
candidate_params = {'radius': 1}
distance = lambda point1, point2: np.linalg.norm(point1 - point2)

disc_generator = Spatial(m = m,
                    voter_dist = sample_uniform_disc, voter_params = voter_params,
                    candidate_dist = sample_uniform_disc, candidate_params = candidate_params,
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
f = 'data/disc_full.npz'
results_list = samples(n_samples, disc_generator, elections_dict, [n], k, dim = 2, filename = f)
result_dict = results_list[0]