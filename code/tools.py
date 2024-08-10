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


def cost(voter_positions, candidate_positions, distance = euclidean_distance):
    cost_sum = 0
    for v in voter_positions:
        for c in candidate_positions:
            cost_sum += distance(v,c)
    return cost_sum

'''
def best_group_cost(voter_positions, candidate_positions, size):
    individual_candidate_costs = np.zeros(len(candidate_positions))

    for i,c in enumerate(candidate_positions):
        cand_cost = cost(voter_positions, [c])
        individual_candidate_costs[i] = cand_cost

    top_cands = np.argsort(individual_candidate_costs)[:size]
    return cost(voter_positions, candidate_positions[top_cands,:])
'''

def best_group_cost(voter_positions, candidate_positions, size):
    diffs = voter_positions[np.newaxis, :, :] - candidate_positions[:, np.newaxis, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
    costs = np.sum(distances, axis=1)
        
    top_cands = np.argsort(costs)[:size]
    return cost(voter_positions, candidate_positions[top_cands,:])


def representativeness(voter_positions, candidate_positions, groups, winners):
    n_voters = len(voter_positions)
    k = len(winners)
    max_epsilon = 0

    for g in groups:
        G = voter_positions[g,:]
        Rsize = int(len(g)/n_voters * k)
        
        if Rsize != 0:
            cost1 = best_group_cost(G, winners, Rsize)
            cost2 = best_group_cost(G, candidate_positions, Rsize)
            eps = np.abs(cost1 - cost2)/Rsize

            if eps > max_epsilon:
                max_epsilon = eps

    return max_epsilon


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