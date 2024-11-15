import numpy as np


def euclidean_distance(x,y):
    """Euclidean Distance between points x and y
    Args:
        x (np.ndarray): point in euclidean space
        y (np.ndarray): point in euclidean space

    Returns:
        (float): computed distance
    """
    return np.linalg.norm(x - y, ord = 2)


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
    """
    Removes a list or array of candidates from the given preference profile,
    and returns the modified profile.
    
    Args:
        profile (np.ndarray): (candidates x voters) Preference profile matrix.
        candidates (list[int] OR np.ndarray): Candidates to remove from the profile.
    
    Returns:
        (np.ndarray): New preference profile with candidates removed.
    """
    
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

# voter_positions is generally voters within a bloc
# CHECKED!
def cost(voter_positions, candidate_positions, distance = euclidean_distance):
    """ 
    With some distance function, computes the sum of distances 
    distance(voter, candidate) for every 
    voter candidate pair in the input list of voter and candidate positions.

    Args:
        voter_positions (np.ndarray): (n x d) Array of voter positions in a metric space.
        candidate_positions (np.ndarray): (m x d) Array of candidate positions in a metric space.
        distance (callable, optional): Callable distance function which should
            take as input two d dimensional vectors and output a real number, 
            defaults to euclidean_distance (see euclidean_distance() above for a reference format).

    Returns:
        float: Sum of distances (cost).
    """
    cost_sum = 0
    for v in voter_positions:
        for c in candidate_positions:
            cost_sum += distance(v,c)
    return cost_sum

# Change to 'euclidean_costs'
# CHECKED!
def costs(voter_positions, candidate_positions):
    """
    Given a set of voter and candidate positions, returns an
    array of costs where each entry is the sum of 
    distances from each voter to a single candidate. For example, 
    for the candidate described by row i in candidate_positions, the sum of
    distances from every voter to that candidate will be output in index i
    of the returned array.
    
    NOTE: This optimized to be efficient, and therefore 
        only uses euclidean distance for now. 

    Args:
        voter_positions (np.ndarray): (n x d) Array of voter positions in a metric space.
        candidate_positions (np.ndarray): (m x d) Array of candidate positions in a metric space.

    Returns:
        (np.ndarray): Array of distances from voters to candidates.
    """
    # cost to each candidate:
    diffs = voter_positions[np.newaxis, :, :] - candidate_positions[:, np.newaxis, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
    cost_array = np.sum(distances, axis=1)
    return cost_array


def voter_costs(voter_positions, candidate_positions):
    """
    Given a set of voter and candidate positions, returns an
    array of costs where each entry is the sum of 
    distances from each candidate to a single voter. For example, 
    for the voter described by row i in voter_positions, the sum of
    distances from every candidate to that voter will be output in index i
    of the returned array.
    
    NOTE: This optimized to be efficient, and therefore 
        only uses euclidean distance for now. 

    Args:
        voter_positions (np.ndarray): (n x d) Array of voter positions in a metric space.
        candidate_positions (np.ndarray): (m x d) Array of candidate positions in a metric space.

    Returns:
        (np.ndarray): Array of distances from candidates to voters.
    """
    # cost to each voter of summed over every candidate:
    diffs = voter_positions[np.newaxis, :, :] - candidate_positions[:, np.newaxis, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
    cost_array = np.sum(distances, axis=0)
    return cost_array

# Make cost_array function instead

def best_group_cost(voter_positions, candidate_positions, size):
    """
    Find the cost of voters to some best subset of candidates
    from candidate_positions with a given size. For example, 
    if size = k, then this will output the sum of distances from each 
    voter to the size k subset of candidates which will minimize
    the sum overall.  

    Args:
        voter_positions (np.ndarray): (n x d) Array of voter positions in a metric space.
        candidate_positions (np.ndarray): (m x d) Array of candidate positions in a metric space.
        size (int): Size required for the best group / subset of candidates. 

    Returns:
        float: Sum of distances (cost).
    """
    if size > len(candidate_positions):
        raise ValueError('Requested size is too large!')
    
    cost_array = costs(voter_positions, candidate_positions)
    return np.sum(np.sort(cost_array)[:size])


def group_representation(voter_positions, candidate_positions, 
                         voter_labels, winner_positions, group_label, size = None):
    """
    Computes the group inefficiency score as the cost ratio between 
    best group among the winners and the best group among all candidates within
    candidate positions.
    
    Optional: Instead of using proportionally sized representative sets, set 
            size = k to enforce that voters are represented by a constant size k set. 
    
    Args:
        voter_positions (np.ndarray): (n x d) Array of voter positions in a metric space.
        candidate_positions (np.ndarray): (m x d) Array of candidate positions in a metric space.
        voter_labels (np.ndarray[int]): Integer array where index i gives 
                                        the group membership of voter i.
        winners (np.ndarray): (k x d) Array of winning candidate positions. 
        group_label (int): Group label to compute score for. 
        size (int, optional): Pre-defined constant size of the representative set
            for input voters. Defaults to None, in which case size is computed 
            proportional to the size of the input set of voters *In most cases 
            we'll default to this!*.

    Returns:
        float: Group inefficiency score. 
    """
    n_voters = len(voter_positions)
    k = len(winner_positions)
    
    voter_labels = np.array([0,1,2,1,0,1,1,2])
    G = voter_positions[voter_labels == group_label, :]
    
    
    # confusing
    group = [i for i in range(len(voter_labels)) if voter_labels[i] == group_label]
    if size is None:
        # Proportional sizing!
        Rsize = int(len(group)/n_voters * k)
    else:
        Rsize = size
    
    if Rsize != 0:
        G = voter_positions[group, :]
        # find the best group cost among winners and among all candidates
        cost1 = best_group_cost(G, winner_positions, Rsize)
        cost2 = best_group_cost(G, candidate_positions, Rsize)
        # and return the ratio!
        return cost1/cost2
    else:
        return 0
   
# WIP 
def max_group_representation(voter_positions, candidate_positions, 
                             voter_labels, winners, size = None):
    """
    Computes the maximum group inefficiency score among all known groups.
    
    Optional: Instead of using proportionally sized representative sets, set 
            size = k to enforce that voters are represented by a constant size k set. 
    
    Args:
        voter_positions (np.ndarray): (n x d) Array of voter positions in a metric space.
        candidate_positions (np.ndarray): (m x d) Array of candidate positions in a metric space.
        voter_labels (np.ndarray[int]): Integer array where index i gives 
                                        the group membership of voter i.
        winners (np.ndarray): (k x d) Array of winning candidate positions. 
        group_label (int): Group label to compute score for. 
        size (int, optional): Pre-defined constant size of the representative set
            for input voters. Defaults to None, in which case size is computed 
            proportional to the size of the input set of voters *In most cases 
            we'll default to this!*.

    Returns:
        float: Maximum group inefficiency score. 
    """
    group_labels = np.unique(voter_labels) 
    alpha = 0
    for g in group_labels:
        g_alpha = group_representation(voter_positions, candidate_positions, 
                                       voter_labels, winners, g, size)
        if g_alpha > alpha:
            alpha = g_alpha
            
    return alpha
        


def qmin_cost(voter_positions, winner_positions, q):
    """
    Sum of distances from every voter to their qth closest candidate from the 
    set of winning candidates. 

    Args:
        voter_positions (np.ndarray): (n x d) Array of voter positions in a metric space.
        candidate_positions (np.ndarray): (k x d) Array of winning candidate positions in a metric space.
        q (int): Value of q for q-min.

    Returns:
        float: Sum of qmin distances for each voter (qmin cost).
    """
    diffs = voter_positions[np.newaxis, :, :] - winner_positions[:, np.newaxis, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
    distance_sort = np.sort(distances, axis = 0)
    return np.sum(distance_sort[q,:])




'''
# Another notion of cost I was experimenting with:
def worst_group_cost(voter_positions, candidate_positions, size, k):
    cost_array = costs(voter_positions, candidate_positions)
    worst_cands = np.argsort(cost_array)[::-1][k-size:k]
    return np.sum(cost_array[worst_cands])
'''