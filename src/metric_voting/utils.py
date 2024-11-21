import numpy as np
from typing import Callable


# Peter Note: CHECKED!
def euclidean_distance(x, y):
    """Euclidean Distance between points x and y
    Args:
        x (np.ndarray): point in euclidean space
        y (np.ndarray): point in euclidean space

    Returns:
        (float): computed distance
    """
    return np.linalg.norm(x - y, ord=2)


def cost_array(voter_positions, candidate_positions, distance=euclidean_distance):
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
            dists[i, j] = distance(voter_positions[j], candidate_positions[i])
    return dists


def euclidean_cost_array(voter_positions, candidate_positions):
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


def cost(dists):
    """
    Given an (m x n) cost array storing all distances from each voter to
    each candidate, finds the total cost by summing all entries.

    Args:
        dists (np.ndarray): (m x n) Array of distances from candidates to voters.

    Returns:
        float: Sum of distances (cost).
    """
    return np.sum(dists)


def candidate_costs(dists):
    """
    Given an (m x n) cost array storing all distances from each voter to
    each candidate, finds the total cost for each candidate
    summed along all voters.

    Args:
        dists (np.ndarray): (m x n) Array of distances from candidates to voters.

    Returns:
        (np.ndarray): Length m array of candidate distances.
    """
    candidate_dists = np.sum(dists, axis=1)
    return candidate_dists


def voter_costs(dists):
    """
    Given an (m x n) cost array storing all distances from each voter to
    each candidate, finds the total cost for each candidate
    summed along all voters.

    Args:
        dists (np.ndarray): (m x n) Array of distances from candidates to voters.

    Returns:
        (np.ndarray): Array of distances from candidates to voters.
    """
    voter_dists = np.sum(dists, axis=0)
    return voter_dists


def proportional_assignment_cost(dists, size):
    """
    Find the cost of voters to some best subset of candidates
    with a given size.

    Args:
        dists (np.ndarray): (m x n) Array of distances from candidates to voters.
        size (int): Size required for the best group / subset of candidates.

    Returns:
        float: Sum of distances (cost).
    """
    if size > len(dists):
        raise ValueError("Requested size is too large!")

    candidate_dists = candidate_costs(dists)
    return np.sum(np.sort(candidate_dists)[:size])


def group_inefficiency(dists, winner_indices, voter_labels, bloc_label, size=None):
    """
    Computes the group inefficiency score as the cost ratio between
    best subset of winners, and the best subset of all candidates.

    Optional: Instead of using proportionally sized representative sets, set
            the size parameter to enforce that voters are represented by a constant size set.

    Args:
        dists (np.ndarray): (m x n) Array of distances from candiates to voters.
        winner_indices (np.ndarray[int]): Length k array of winning candidate indices.
        voter_labels (np.ndarray[int]): Integer array where index i gives
                                        the group membership of voter i.
        bloc_label (int): Bloc label to compute score for.
        size (int, optional): Pre-defined constant size of the representative set
            for input voters. Defaults to None, in which case size is computed
            proportional to the size of the input set of voters *In most cases
            we'll default to this!*.

    Returns:
        float: Group inefficiency score.
    """
    m, n = dists.shape
    k = len(winner_indices)

    if size is None:
        # Proportional sizing!
        bloc_size = np.sum(voter_labels == bloc_label)
        size = int(bloc_size / n * k)

    if size != 0:
        bloc_dists = dists[:, voter_labels == bloc_label]
        cost1 = proportional_assignment_cost(bloc_dists[winner_indices, :], size)
        cost2 = proportional_assignment_cost(bloc_dists, size)
        return cost1 / cost2
    else:
        return 0


def random_group_inefficiency(dists, winner_indices, size=None):
    """
    For a randomly selected bloc of voters, computes the group
    inefficiency score.
    """
    m, n = dists.shape
    k = len(winner_indices)
    min_size = int((n / k) + 1)
    bloc_size = np.random.randint(min_size, n + 1)
    random_bloc = np.random.choice(n, bloc_size, replace=False)
    voter_labels = np.zeros(n)
    voter_labels[random_bloc] = 1
    return group_inefficiency(dists, winner_indices, voter_labels, 1, size), random_bloc


def greedy_group_inefficiency(dists, winner_indices, size=None):
    """
    For a greedily selected bloc of voters, computes the group
    inefficiency score.
    """
    m, n = dists.shape
    k = len(winner_indices)
    min_size = int((n / k) + 1)
    voter_labels = np.zeros(n)

    winner_set_dists = np.array([np.sum(dists[winner_indices, i]) for i in range(n)])
    candidate_set_dists = np.array([np.sum(np.sort(dists[:, i])[:k]) for i in range(n)])

    # winner_set_dists = np.array([np.min(dists[winner_indices,i]) for i in range(n)])
    # candidate_set_dists = np.array([np.min(dists[:,i]) for i in range(n)])

    greedy_scores = winner_set_dists / candidate_set_dists
    greedy_order = np.argsort(greedy_scores)[::-1]

    voter_labels[greedy_order[:min_size]] = 1
    ineff = group_inefficiency(dists, winner_indices, voter_labels, 1, size)
    for i in greedy_order[min_size:]:
        voter_labels[i] = 1
        new_ineff = group_inefficiency(dists, winner_indices, voter_labels, 1, size)
        if new_ineff > ineff:
            ineff = new_ineff
        else:
            voter_labels[i] = 0

    greedy_bloc = np.where(voter_labels == 1)[0]
    return ineff, greedy_bloc


def random_greedy_group_inefficiency(dists, winner_indices, size=None):
    """
    For a greedily selected bloc of voters, computes the group
    inefficiency score.
    """
    m, n = dists.shape
    k = len(winner_indices)
    min_size = int((n / k) + 1)
    voter_labels = np.zeros(n)

    winner_set_dists = np.array([np.sum(dists[winner_indices, i]) for i in range(n)])
    candidate_set_dists = np.array([np.sum(np.sort(dists[:, i])[:k]) for i in range(n)])

    # winner_set_dists = np.array([np.min(dists[winner_indices,i]) for i in range(n)])
    # candidate_set_dists = np.array([np.min(dists[:,i]) for i in range(n)])

    greedy_scores = winner_set_dists / candidate_set_dists

    # Sort with noise:
    greedy_scores = greedy_scores + np.random.normal(0, np.std(greedy_scores), n)
    greedy_order = np.argsort(greedy_scores)[::-1]

    voter_labels[greedy_order[:min_size]] = 1
    ineff = group_inefficiency(dists, winner_indices, voter_labels, 1, size)
    for i in greedy_order[min_size:]:
        voter_labels[i] = 1
        new_ineff = group_inefficiency(dists, winner_indices, voter_labels, 1, size)
        if new_ineff > ineff:
            ineff = new_ineff
        else:
            voter_labels[i] = 0

    greedy_bloc = np.where(voter_labels == 1)[0]
    return ineff, greedy_bloc


'''
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
'''


def borda_matrix(
    profile, scoring_scheme: Callable[[int, int], float] = lambda x, y: x - y
):
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


def remove_candidates(profile, candidates):
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


# WIP
# Peter did not check this
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


"""
# Another notion of cost I was experimenting with:
def worst_group_cost(voter_positions, candidate_positions, size, k):
    cost_array = costs(voter_positions, candidate_positions)
    worst_cands = np.argsort(cost_array)[::-1][k-size:k]
    return np.sum(cost_array[worst_cands])
"""
