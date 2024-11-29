import numpy as np
import math
from numpy.typing import NDArray
from typing import Callable, Tuple, Any, Optional, Union


def euclidean_distance(x, y):
    """Euclidean Distance between points x and y
    Args:
        x (np.ndarray): point in euclidean space
        y (np.ndarray): point in euclidean space

    Returns:
        (float): computed distance
    """
    return np.linalg.norm(x - y, ord=2)


def cost_array(
        voter_positions: NDArray, 
        candidate_positions: NDArray, 
        distance_fn: Callable = euclidean_distance
) -> NDArray:
    """
    Given a set of voter and candidate positions along with a distance function,
    returns an (m x n) array with each entry i,j storing the
    distance from candidate i to voter j.

    Args:
        voter_positions (np.ndarray): (n x d) Array of voter positions in a metric space.
        candidate_positions (np.ndarray): (m x d) Array of candidate positions in a metric space.
        distance (callable, optional): Callable distance function which should
            take as input two d dimensional vectors and output a real number,
            defaults to euclidean_distance (see euclidean_distance() above for a reference format).

    Returns:
        (np.ndarray): Size (m x n) array of distances from voters to candidates.
    """

    dists = np.zeros((len(candidate_positions), len(voter_positions)))
    for i in range(len(candidate_positions)):
        for j in range(len(voter_positions)):
            dists[i, j] = distance_fn(voter_positions[j], candidate_positions[i])
    return dists


def euclidean_cost_array(
    voter_positions: NDArray, 
    candidate_positions: NDArray,
) -> NDArray:
    """
    Given a set of voter and candidate positions, returns an
    (m x n) array with each entry i,j storing the
    distance from candidate i to voter j.

    NOTE: This is an optimized version of cost_array()
        which assumes distance is euclidean.

    Args:
        voter_positions (np.ndarray): (n x d) Array of voter positions in a metric space.
        candidate_positions (np.ndarray): (m x d) Array of candidate positions in a metric space.

    Returns:
        (np.ndarray): Size (m x n) array of distances from voters to candidates.
    """
    diffs = voter_positions[np.newaxis, :, :] - candidate_positions[:, np.newaxis, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    return dists


def cost(cst_array: NDArray) -> float:
    """
    Given an (m x n) cost array storing all distances from each voter to
    each candidate, finds the total cost by summing all entries.

    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 

    Returns:
        float: Sum of distances (cost).
    """
    return np.sum(cst_array)


def candidate_costs(cst_array: NDArray) -> NDArray:
    """
    Given an (m x n) cost array storing all distances from each voter to
    each candidate, finds the total cost for each candidate
    summed along all voters.

    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 

    Returns:
        (np.ndarray): Length m array of distance sums for candidates.
    """
    candidate_csts = np.sum(cst_array, axis=1)
    return candidate_csts


def voter_costs(cst_array: NDArray) -> NDArray:
    """
    Given an (m x n) cost array storing all distances from each voter to
    each candidate, finds the total cost for each voter
    summed across all candidates.

    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 

    Returns:
        (np.ndarray): Length n array of distance sums for voters.
    """
    voter_csts = np.sum(cst_array, axis=0)
    return voter_csts


def proportional_assignment_cost(cst_array: NDArray, size: int) -> float:
    """
    Given voters and candidates in an input cost array, finds the lowest cost 
    assignment of voters to a subset of candidates with a given size. 

    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        size (int): Size required for the selected subset of candidates.

    Returns:
        float: Sum of distances (cost).
    """
    if size > len(cst_array):
        raise ValueError("Requested size is too large!")

    candidate_csts = candidate_costs(cst_array)
    return np.sum(np.sort(candidate_csts)[:size])


def group_inefficiency(
    cst_array: NDArray, 
    winner_indices: NDArray, 
    voter_labels: NDArray, 
    bloc_label: int, 
    size: Optional[int] = None
) -> float:
    """
    Computes the group inefficiency score as the cost ratio between
    best subset of winners, and the best subset of all candidates.

    Optional: Instead of using proportionally sized representative sets, set
            the size parameter to enforce that voters are represented by a constant size set.

    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        winner_indices (np.ndarray[int]): Length k array of winning candidate indices.
        voter_labels (np.ndarray[int]): Integer array where index i gives
                                        the bloc label of voter i.
        bloc_label (int): Selected bloc label to compute score for.
        size (int, optional): Pre-defined constant size of the representative set
            for input voters. Defaults to None, in which case size is computed
            proportional to the size of the input set of voters *In most cases
            we'll default to this!*.

    Returns:
        float: Group inefficiency score.
    """
    _, n = cst_array.shape
    k = len(winner_indices)

    if size is None:
        # Proportional sizing!
        bloc_size = np.sum(voter_labels == bloc_label)
        size = int(bloc_size / n * k)

    if size != 0:
        bloc_dists = cst_array[:, voter_labels == bloc_label]
        cost1 = proportional_assignment_cost(bloc_dists[winner_indices, :], size)
        cost2 = proportional_assignment_cost(bloc_dists, size)
        return cost1 / cost2

    return 0


# WIP
def generate_uniform_random_voter_bloc(n, k):
    min_size = int(np.ceil(n / k))
    bloc_size = np.random.randint(min_size, n + 1)
    random_bloc = np.random.choice(n, bloc_size, replace=False)
    voter_labels = np.zeros(n)
    voter_labels[random_bloc] = 1
    return random_bloc, voter_labels


def generate_weighted_random_voter_bloc(n, k, t, weights):
    min_size = math.ceil(n * t / k)
    max_size = math.ceil(n * (t + 1) / k) - 1
    bloc_size = np.random.randint(min_size, max_size)
    random_bloc = np.random.choice(n, bloc_size, replace=False, p = weights)
    voter_labels = np.zeros(n)
    voter_labels[random_bloc] = 1
    return random_bloc, voter_labels



def random_group_inefficiency(
    cost_arr: NDArray, 
    winner_indices: NDArray, 
) -> Tuple[float, NDArray]:
    """
    For a randomly selected bloc of voters, computes the group
    inefficiency score.
    """
    _, n = cost_arr.shape
    k = len(winner_indices)
    random_bloc, voter_labels = generate_uniform_random_voter_bloc(n, k)
    return group_inefficiency(cost_arr, winner_indices, voter_labels, bloc_label=1, size=None), random_bloc


def weighted_random_group_inefficiency(
    cost_arr: NDArray, 
    winner_indices: NDArray, 
    t: Optional[int] = 1,
) -> Tuple[float, NDArray]:
    """
    For a randomly selected bloc of voters, computes the group
    inefficiency score.
    """
    _, n = cost_arr.shape
    k = len(winner_indices)
    
    # Get cost to winners for all voters (1,n) size
    winner_set_cost_arr = np.sum(np.sort(cost_arr[winner_indices, :], axis = 0)[:t, :], axis = 0)

    # Cost of best k candidates for each voter (1,n) size
    candidate_set_cost_arr = np.sum(np.sort(cost_arr, axis = 0)[:t, :], axis = 0)

    # Inefficiency by voter
    greedy_scores = winner_set_cost_arr / candidate_set_cost_arr
    greedy_scores /= np.sum(greedy_scores)
    
    random_bloc, voter_labels = generate_weighted_random_voter_bloc(n, k, t = t, weights = greedy_scores)
    return group_inefficiency(cost_arr, winner_indices, voter_labels, bloc_label=1, size=None), random_bloc


def greedy_group_inefficiency(
    cost_arr: NDArray, 
    winner_indices: NDArray, 
    size: Optional[int] = None
) -> Tuple[float, NDArray]:
    """
    For a greedily selected bloc of voters, computes the group
    inefficiency score.
    """
    _, n = cost_arr.shape
    k = len(winner_indices)
    min_size = int((n / k) + 1)
    voter_labels = np.zeros(n)

    # Get cost to winners for all voters (1,n) size
    winner_set_cost_arr = np.sum(cost_arr[winner_indices], axis = 0)

    # Cost of best k candidates for each voter (1,n) size
    candidate_set_cost_arr = np.sum(np.sort(cost_arr, axis = 0)[:k, :], axis = 0)

    # Inefficiency by voter
    greedy_scores = winner_set_cost_arr / candidate_set_cost_arr
    greedy_order = np.flip(np.argsort(greedy_scores))

    voter_labels[greedy_order[:min_size]] = 1
    ineff = group_inefficiency(cost_arr, winner_indices, voter_labels, 1, size)


    # Peter Note: This needs another loop. Could add later voter that then makes earlier
    # voter a good add to the inefficiency
    new_changes = True
    while new_changes:
        new_changes = False
        for i in greedy_order[min_size:]:
            new_ineff = 0

            if voter_labels[i] != 1:
                voter_labels[i] = 1
                new_ineff = group_inefficiency(cost_arr, winner_indices, voter_labels, 1, size)

            if new_ineff > ineff:
                ineff = new_ineff
                new_changes = True
                
            else:
                voter_labels[i] = 0

    greedy_bloc = np.where(voter_labels == 1)[0]
    return ineff, greedy_bloc


def random_greedy_group_inefficiency(cost_arr, winner_indices, size=None):
    """
    For a greedily selected bloc of voters, computes the group
    inefficiency score.
    """
    _, n = cost_arr.shape
    k = len(winner_indices)
    min_size = int((n / k) + 1)
    voter_labels = np.zeros(n)

    # Get cost to winners for all voters (1,n) size
    winner_set_cost_arr = np.sum(cost_arr[winner_indices], axis = 0)

    # Cost of best k candidates for each voter (1,n) size
    candidate_set_cost_arr = np.sum(np.sort(cost_arr, axis = 0)[:k, :], axis = 0)

    # Sort with noise:
    greedy_scores = winner_set_cost_arr / candidate_set_cost_arr

    # Change the standard dev to be the std_dev of pairwise score distances
    greedy_scores = greedy_scores + np.random.normal(0, np.std(greedy_scores), n)
    greedy_order = np.argsort(greedy_scores)[::-1]

    voter_labels[greedy_order[:min_size]] = 1
    ineff = group_inefficiency(cost_arr, winner_indices, voter_labels, 1, size)
    
    # Also needs another loop
    for i in greedy_order[min_size:]:
        voter_labels[i] = 1
        new_ineff = group_inefficiency(cost_arr, winner_indices, voter_labels, 1, size)
        if new_ineff > ineff:
            ineff = new_ineff
        else:
            voter_labels[i] = 0

    greedy_bloc = np.where(voter_labels == 1)[0]
    return ineff, greedy_bloc



def borda_matrix(
    profile: NDArray, scoring_scheme: Callable[[int, int], float] = lambda x, y: x - y
) -> NDArray:
    """
    Computes a borda matrix given an input preference profile.
    Specifically, for an (m x n) profile the borda matrix at
    entry [i,j] records the borda score of candidate i for voter j.

    Args:
        profile (np.ndarray): (candidates x voters) Preference profile matrix.

    Returns:
        (np.ndarray): Computed (m x n) borda matrix.
    """
    m, n = profile.shape
    B = np.zeros((m, n))
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            B[profile[i, j], j] = scoring_scheme(m, i)

    return B


def remove_candidates(profile: NDArray, candidates: Union[list, NDArray]) -> NDArray:
    """
    Removes a list or array of candidates from the given preference profile,
    and returns the modified profile.

    Args:
        profile (np.ndarray): (candidates x voters) Preference profile matrix.
        candidates (list[int] OR np.ndarray): Candidates to remove from the profile.

    Returns:
        (np.ndarray): New (m - len(candidates) x n) preference profile with candidates removed.
    """

    return np.array([row[~np.isin(row, candidates)] for row in profile.T]).T


def is_complete_ranking(ranking: NDArray) -> bool:
    """
    Checks if the input ranking is a complete ranking of the same m candidates.

    Args:
        ranking (np.ndarray): (m) Array of candidate indices.

    Returns:
        (bool): True if the ranking is complete, False otherwise.
    """
    return np.all(np.isin(np.arange(len(ranking)), np.unique(ranking)))


def approve_profile(profile: NDArray) -> bool:
    """
    Checks if the input profile is an ranked preference profile in 
    correct form. Specifically, for our simplified models, this means 
    a complete ranking of the same m candidates for each voter. 

    Args:
        profile (np.ndarray): (candidates x voters) Preference profile matrix.

    Returns:
        (bool): True if the profile is approved, False otherwise.
    """
    return np.all(np.apply_along_axis(is_complete_ranking, axis = 0, arr = profile))


def uniform_profile(n: int, m: int) -> NDArray:
    """
    Generates a uniform profile with m candidates and n voters, where each voter
    ranks the candidates in a random order.

    Args:
        n (int): Number of voters.
        m (int): Number of candidates.

    Returns:
        (np.ndarray): Generated preference profile.
    """
    return np.array([np.random.permutation(m) for _ in range(n)]).T



'''
A bit of older code that may be useful later...

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
        g_alpha = group_representation(
            voter_positions, candidate_positions, voter_labels, winners, g, size
        )
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
    distances = np.sqrt(np.sum(diffs**2, axis=-1))
    distance_sort = np.sort(distances, axis=0)
    return np.sum(distance_sort[q, :])

# Another notion of cost I was experimenting with:
def worst_group_cost(voter_positions, candidate_positions, size, k):
    cost_array = costs(voter_positions, candidate_positions)
    worst_cands = np.argsort(cost_array)[::-1][k-size:k]
    return np.sum(cost_array[worst_cands])
'''
