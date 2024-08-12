import numpy as np


def euclidean_distance(x,y):
    """Euclidean Distance between points x and y
    Args:
        x (np.ndarray): point in euclidean space
        y (np.ndarray): point in euclidean space

    Returns:
        (float): computed distance
    """
    return np.linalg.norm(x - y)


def borda_matrix(profile):
    """
    Computes a borda matrix given an input preference profile. 
    Specifically, for an (m x n) profile the borda matrix at 
    entry [i,j] records the borda score of candidate i for voter j.
    
    Args:
        profile (np.ndarray): (candidates x voters) Preference profile matrix.
    
    Returns:
        (np.ndarray): Computed borda matrix.
    """
    m,n = profile.shape
    B = np.zeros((m,n))
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            B[profile[i,j], j] = (m - 1) - i
            
    return B


def remove_candidates(profile, candidates):
    # Convert candidates to a set for faster lookup
    candidates_set = set(candidates)
    
    # Determine the size of the new profile
    remaining_candidates = profile.shape[0] - len(candidates)
    new_profile = np.zeros((remaining_candidates, profile.shape[1]), dtype=int)
    
    for col in range(profile.shape[1]):
        # Filter out candidates that are in the candidates_set
        new_rank = [i for i in profile[:,col] if i not in candidates_set]
        # Assign the new rank to the corresponding column in the new profile
        new_profile[:,col] = new_rank
    
    return new_profile


def cost(voter_positions, candidate_positions, distance = euclidean_distance):
    cost_sum = 0
    for v in voter_positions:
        for c in candidate_positions:
            cost_sum += distance(v,c)
    return cost_sum


def costs(voter_positions, candidate_positions):
    # cost to each candidate:
    diffs = voter_positions[np.newaxis, :, :] - candidate_positions[:, np.newaxis, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
    cost_array = np.sum(distances, axis=1)
    return cost_array


def voter_costs(voter_positions, candidate_positions):
    # cost to each voter of summed over every candidate:
    diffs = voter_positions[np.newaxis, :, :] - candidate_positions[:, np.newaxis, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
    cost_array = np.sum(distances, axis=0)
    return cost_array



def best_group_cost(voter_positions, candidate_positions, size):
    cost_array = costs(voter_positions, candidate_positions)
    best_cands = np.argsort(cost_array)[:size]
    return np.sum(cost_array[best_cands])


def worst_group_cost(voter_positions, candidate_positions, size, k):
    cost_array = costs(voter_positions, candidate_positions)
    worst_cands = np.argsort(cost_array)[::-1][k-size:k]
    return np.sum(cost_array[worst_cands])


def group_representation(voter_positions, candidate_positions, voter_labels, winners, group_label, size = None):
    n_voters = len(voter_positions)
    k = len(winners)
    group = [i for i in range(len(voter_labels)) if voter_labels[i] == group_label]
    if size is None:
        Rsize = int(len(group)/n_voters * k)
    else:
        Rsize = size
    
    if Rsize != 0:
        G = voter_positions[group, :]
        cost1 = best_group_cost(G, winners, Rsize)
        cost2 = best_group_cost(G, candidate_positions, Rsize)
        return cost1/cost2
    else:
        return 0
    
def max_group_representation(voter_positions, candidate_positions, voter_labels, winners, size = None):
    group_labels = np.unique(voter_labels) 
    alpha = 0
    for g in group_labels:
        g_alpha = group_representation(voter_positions, candidate_positions, voter_labels, winners, g, size)
        


def representativeness(voter_positions, candidate_positions, voter_labels, winners, sizes = None):
    n_voters = len(voter_positions)
    groups = [[j for j in range(len(voter_labels)) if voter_labels[j] == i] 
              for i in np.unique(voter_labels)]
    k = len(winners)
    max_epsilon = 0

    for g in groups:
        G = voter_positions[g,:]
        if sizes is None:
            Rsize = int(len(g)/n_voters * k)
        else:
            Rsize = sizes
        
        if Rsize != 0:
            cost1 = best_group_cost(G, winners, Rsize)
            cost2 = best_group_cost(G, candidate_positions, Rsize)
            cost3 = worst_group_cost(G, candidate_positions, Rsize, k)

            #eps = np.abs(cost1 - cost2)/Rsize
            numerator = np.abs(cost1 - cost2)
            denominator = np.abs(cost3 - cost2)

            if np.isclose(numerator, 0, atol = 1e-8) and np.isclose(denominator, 0, atol = 1e-8):
                eps = 0
            else:
                eps = np.abs(cost1 - cost2)/np.abs(cost3 - cost2)

            if eps > max_epsilon:
                max_epsilon = eps

    return max_epsilon


def representativeness_ratio(voter_positions, candidate_positions, voter_labels, winners, sizes = None):
    n_voters = len(voter_positions)
    groups = [[j for j in range(len(voter_labels)) if voter_labels[j] == i] 
              for i in np.unique(voter_labels)]
    k = len(winners)
    max_epsilon = 0

    for g in groups:
        G = voter_positions[g,:]
        if sizes is None:
            Rsize = int(len(g)/n_voters * k)
        else:
            Rsize = sizes
        
        if Rsize != 0:
            cost1 = best_group_cost(G, winners, Rsize)
            cost2 = best_group_cost(G, candidate_positions, Rsize)
            #cost3 = worst_group_cost(G, candidate_positions, Rsize, k)
            eps = cost1/cost2

            if eps > max_epsilon:
                max_epsilon = eps

    return max_epsilon
    