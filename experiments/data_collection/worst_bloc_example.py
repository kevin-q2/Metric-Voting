import numpy as np
import pandas as pd
import math
from metric_voting import *

np.random.seed(918717)
filename = 'data/2bloc_worst_example3.npz'

# Choose number of voters n and the number of candidates m
n = 1000
m = 1000

# Choose the number of winners for the election
k = 4

# Means and standard deviations for each of the two voter distributions
means = [[0, -2], [0, 2]]
stds = [0.5, 0.5]
two_party_G = [500,500]  # Group Sizes

# Create a list of voter parameters -- with each set of parameters being a dict
voter_params = [{'loc': None, 'scale': None, 'size': 2} for _ in range(len(two_party_G))]
for i,mean in enumerate(means):
    voter_params[i]['loc'] = mean

for i,std in enumerate(stds):
    voter_params[i]['scale'] = std
    
# Define the single set of candidate paramters
candidate_params = [{'low': -5, 'high': 5, 'size': 2}]

# define a distance function between voters and candidates
distance = lambda point1, point2: np.linalg.norm(point1 - point2)

# Create the group spatial generator object!
two_party_generator = GroupSpatial(n_voter_groups = 2, n_candidate_groups = 1,
                                   voter_dist_fns = [np.random.normal]*len(two_party_G),
                                   voter_dist_fn_params = voter_params,
                                   candidate_dist_fns = [np.random.uniform],
                                   candidate_dist_fn_params = candidate_params,
                                   distance_fn = distance)


# Now each call .generate() will make a profile with random candidate and voter positions
profile, \
candidate_positions, \
voter_positions, \
candidate_labels, \
voter_labels = two_party_generator.generate(voter_group_sizes = two_party_G, 
                                            candidate_group_sizes = [m])



# Collect results for different blocs and their representatives
n_samples = 100000
result_dict = {'voters' : voter_positions, 'candidates' : candidate_positions}
cst_array = euclidean_cost_array(voter_positions, candidate_positions)
elections_dict = {Bloc:{}, Borda:{}, STV:{'transfer_type' : 'weighted-fractional'}}

for E, params in elections_dict.items():
    e_dict = {}
    name = E.__name__
    winners = E(**params).elect(profile,k)
    winner_mask = np.zeros(m, dtype=bool)
    winner_mask[winners] = True
    e_dict['winners'] = winner_mask
    
    bloc_label = 1
    # single group ineff
    group_mask = voter_labels
    group_reps1 = proportional_assignment(cst_array[winners, :], group_mask, bloc_label, k)
    group_reps2 = proportional_assignment(cst_array[winners, :], group_mask, 1 - bloc_label, k)
    group_rep_mask1 = np.zeros(m, dtype=bool)
    group_rep_mask2 = np.zeros(m, dtype=bool)
    group_rep_mask1[winners[group_reps1]] = True
    group_rep_mask2[winners[group_reps2]] = True
    group_ineff1 = group_inefficiency(cst_array, winners, group_mask, bloc_label)
    group_ineff2 = group_inefficiency(cst_array, winners, group_mask, 1 - bloc_label)
    
    if group_ineff1 > group_ineff2:
        e_dict['group'] = {
            'labels' : group_mask,
            'reps' : group_rep_mask1,
            'ineff' : group_ineff1
        }
    else:
        e_dict['group'] = {
            'labels' : 1 - group_mask,
            'reps' : group_rep_mask2,
            'ineff' : group_ineff2
        }

    # overall group ineff
    overall_mask = np.ones(n, dtype = int)
    overall_reps = proportional_assignment(cst_array[winners, :], overall_mask, bloc_label, k)
    overall_rep_mask = np.zeros(m, dtype=bool)
    overall_rep_mask[winners[overall_reps]] = True
    overall_ineff = group_inefficiency(cst_array, winners, overall_mask, bloc_label)
    e_dict['overall'] = {
        'labels' : overall_mask,
        'reps' : overall_rep_mask,
        'ineff' : overall_ineff
    }

    # heuristic group ineff 
    '''
    smallest_size = math.ceil(n/k)
    if name == 'Borda':
        heuristic = np.argsort(voter_positions[:,1])[::-1][:smallest_size]
        heuristic_mask = np.zeros(n, dtype = int)
        heuristic_mask[heuristic] = 1
        
    elif name == 'STV':
        heuristic = np.argsort(np.abs(voter_positions[:,1]))[:smallest_size]
        heuristic_mask = np.zeros(n, dtype = int)
        heuristic_mask[heuristic] = 1
        
    elif name == 'Bloc':
        heuristic = np.argsort(np.abs(voter_positions[:,1]))[:smallest_size]
        heuristic_mask = np.zeros(n, dtype = int)
        heuristic_mask[heuristic] = 1
        
        #heuristic_bloc = heuristic_group(voter_positions, candidate_positions[winners,:])
        #heuristic_mask = np.zeros(n, dtype = int)
        #heuristic_mask[heuristic_bloc] = 1
    
    else:
        heuristic_mask = group_mask
    '''
        
    heuristic_bloc = heuristic_worst_bloc(cst_array, winners)
    heuristic_mask = np.zeros(n, dtype = int)
    heuristic_mask[heuristic_bloc] = 1
        
    heuristic_reps = proportional_assignment(cst_array[winners, :], heuristic_mask, bloc_label, k)
    heuristic_rep_mask = np.zeros(m, dtype=bool)
    heuristic_rep_mask[winners[heuristic_reps]] = True
    heuristic_ineff = group_inefficiency(cst_array, winners, heuristic_mask, bloc_label)
    e_dict['heuristic'] = {
        'labels' : heuristic_mask,
        'reps' : heuristic_rep_mask,
        'ineff' : heuristic_ineff
    }

    # random group ineff
    (worst_estimate, 
     worst_estimate_bloc) = worst_random_group_inefficiency(
         n_samples = n_samples,
         cost_arr = cst_array,
         winner_indices = winners,
         weights = np.ones(n)/n
     )
    random_mask = np.zeros(n, dtype = int)
    random_mask[worst_estimate_bloc] = 1
    random_reps = proportional_assignment(cst_array[winners, :], random_mask, bloc_label, k)
    random_rep_mask = np.zeros(m, dtype=bool)
    random_rep_mask[winners[random_reps]] = True
    random_ineff = worst_estimate
    e_dict['random'] = {
        'labels' : random_mask,
        'reps' : random_rep_mask,
        'ineff' : random_ineff
    }
    
    result_dict[name] = e_dict
    
    
np.savez_compressed(filename, **result_dict)
    
