import numpy as np
from sklearn.clustering import KMeans
from metric_voting import *


scot_profile = np.load('data/scot-elex/aberdeen_2022_ward11.npy')

m = scot_profile.shape[0]
n = scot_profile.shape[1]
k = 3
d = 8
#candidate_positions = np.random.uniform(0,1, size = (m,d))
#candidate_positions = np.zeros((m,d))
#np.fill_diagonal(candidate_positions,1)
candidate_positions = np.random.normal(loc = -1, scale = 0.05, size = (m,d))
np.fill_diagonal(candidate_positions[:,3:],1)
candidate_positions[0, :3] = [-0.18, -0.30, 1.01]
candidate_positions[1, :3] = [1.52, 0.89, -1.37]
candidate_positions[2, :3] = [-0.47, 0.22, -0.65]
candidate_positions[3, :3] = [-0.63, -0.87, 0.53]
candidate_positions[4, :3] = [0.42, -0.41, -0.32]

voter_dist_fn = np.random.uniform
voter_dist_fn_params = {'low' : -2, 'high' : 2, 'size' : d}
distance = lambda point1, point2: np.linalg.norm(point1 - point2)

ranked_generator = RankedSpatial(
    voter_dist_fn,
    voter_dist_fn_params
)

(profile,
candidate_positions,
voter_positions,
candidate_labels,
voter_labels) = ranked_generator.generate(
    n = n,
    m = m,
    profile = scot_profile,
    candidate_positions = candidate_positions
)

clustering = KMeans(n_clusters = 2).fit(voter_positions)
voter_labels = clustering.labels_

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
f = 'data/scot-elex/aberdeen_2022_ward11_results.npz'

generator_input = [
    {'n': n,
    'm': m,
    'profile': scot_profile,
    'candidate_positions': candidate_positions,
    'voter_labels' : voter_labels
    }
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