import numpy as np
import math
from numpy.typing import NDArray
from typing import Callable, Tuple, Any, Optional, Union, List, Set
from .utils import euclidean_distance, random_voter_bloc


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
        distance_fn (callable, optional): Callable distance function which should
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


def min_assignment(cst_array: NDArray, size: int) -> NDArray:
    """
    Given voters and candidates in an input cost array, finds the lowest cost 
    representative subset of candidates with a given size. 

    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        size (int): Size required for the selected subset of candidates.

    Returns:
        NDArray : Length size array of candidate indices
    """
    if size > len(cst_array):
        raise ValueError("Requested size is too large!")

    candidate_csts = candidate_costs(cst_array)
    return np.argsort(candidate_csts)[:size]


def min_assignment_cost(cst_array: NDArray, size: int) -> float:
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


def proportional_assignment(
    cst_array: NDArray,  
    voter_labels: NDArray, 
    bloc_label: int,
    k : int
) -> NDArray:  
    """
    Given voters and candidates in an input cost array, finds the lowest cost 
    representative subset of candidates with proportionally chosen size. 
    
    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        voter_labels (np.ndarray[int]): Integer array where index i gives
                                        the bloc label of voter i.
        bloc_label (int): Selected bloc label to compute score for.
        k (int): Number of winners.
    
    Returns:
        NDArray : Length size array of candidate indices
    """
    _, n = cst_array.shape
    bloc_size = np.sum(voter_labels == bloc_label)
    bloc_array = cst_array[:, voter_labels == bloc_label]
    size = int(bloc_size / n * k)
    return min_assignment(bloc_array, size)


def proportional_assignment_cost(
    cst_array: NDArray,  
    voter_labels: NDArray, 
    bloc_label: int,
    k : int
) -> float:  
    """
    Given voters and candidates in an input cost array, finds the lowest cost 
    associated with the representative subset of candidates with proportionally chosen size. 
    
    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        voter_labels (np.ndarray[int]): Integer array where index i gives
                                        the bloc label of voter i.
        bloc_label (int): Selected bloc label to compute score for.
        k (int): Number of winners.
    
    Returns:
        float: Sum of distances (cost).
    """
    _, n = cst_array.shape
    bloc_size = np.sum(voter_labels == bloc_label)
    bloc_array = cst_array[:, voter_labels == bloc_label]
    size = int(bloc_size / n * k)
    return min_assignment_cost(bloc_array, size)



def group_inefficiency(
    cst_array: NDArray, 
    winner_indices: NDArray, 
    voter_labels: NDArray, 
    bloc_label: int
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
        
    Returns:
        float: Group inefficiency score.
    """
    k = len(winner_indices)
    cost1 = proportional_assignment_cost(
        cst_array[winner_indices, :],
        voter_labels,
        bloc_label,
        k
    )
    cost2 = proportional_assignment_cost(
        cst_array,
        voter_labels,
        bloc_label,
        k
    )
    
    if cost1 == 0 and cost2 == 0:
        return 0 
    
    return cost1 / cost2


def random_group_inefficiency(
    cost_arr: NDArray, 
    winner_indices: NDArray, 
    t: Optional[int] = 1,
    weights : Optional[NDArray] = None
) -> Tuple[float, NDArray]:
    """
    For a randomly selected bloc of voters with t representatives, computes the group
    inefficiency score. Each voter is selected with probabilities proportional to their
    weights. By default, the weights are computed using a greedy heuristic. Otherwise, 
    they may be passed in as input.
    
    Args:
        cost_arr (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        winner_indices (np.ndarray[int]): Length k array of winning candidate indices.
        t (int, optional): Number of representatives that the voter bloc 'deserves.'
            Defaults to 1.
        weights (np.ndarray, optional): Voter probabilities of selection. Defaults to None, 
            in which case a greedy heuristic is used. 
    """
    _, n = cost_arr.shape
    k = len(winner_indices)
    
    if weights is None:
        # Get cost to winners for all voters
        winner_set_cost_arr = np.sum(
            np.sort(cost_arr[winner_indices, :], axis = 0)[:t, :], axis = 0
        )

        # Cost of best k candidates for each voter
        candidate_set_cost_arr = np.sum(
            np.sort(cost_arr, axis = 0)[:t, :], axis = 0
        )

        # Greedy estimate / inefficiency heuristic for voters
        greedy_scores = (winner_set_cost_arr / candidate_set_cost_arr)
        weights = greedy_scores / np.sum(greedy_scores)
    
    random_bloc = random_voter_bloc(n, k, t = t, weights = weights)
    random_bloc_labels = np.zeros(n)
    random_bloc_labels[random_bloc] = 1
    return (
        group_inefficiency(cost_arr, winner_indices, random_bloc_labels, bloc_label=1), 
        random_bloc
    )
    
    
def worst_random_group_inefficiency(
    n_samples : int,
    cost_arr: NDArray, 
    winner_indices: NDArray, 
    weights : Optional[NDArray] = None
) -> Tuple[float, NDArray]:
    """
    Over a series of samples, computes the group inefficiency score for a random group, 
    and outputs the sample that gave the worst ineffciency. 
    
    Args:
        n_samples (int): Number of samples to take.
        cost_arr (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        winner_indices (np.ndarray[int]): Length k array of winning candidate indices.
        weights (np.ndarray, optional): Voter probabilities of selection. Defaults to None, 
            in which case a greedy heuristic is used. 
    """
    worst_score = 0
    worst_bloc = None 
    
    for _ in range(n_samples):
        rand_t = np.random.randint(1, len(winner_indices) + 1)
        score, bloc = random_group_inefficiency(
            cost_arr, 
            winner_indices, 
            t = rand_t, 
            weights = weights
        )
        if score > worst_score:
            worst_score = score
            worst_bloc = bloc
    
    return worst_score, worst_bloc



def q_costs(
    q : int,
    cst_array : NDArray
):
    """
    Given a set of voter and candidate positions along with a distance function,
    returns a length n array with each entry i storing the
    distance from voter i to its qth closest candidate.

    Args:
        q (int): qth closest candidate to compute distance for. 
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 

    Returns:
        (np.ndarray): Size n array of distances from voters to candidates.
    """
    return np.sort(cst_array, axis = 0)[q, :]


def q_cost_array(
    q : int,
    cst_array : NDArray,
    candidate_subsets : List[Set[int]]
):
    """
    Given a list of candidate subsets, computes the q cost array for each subset.

    Args:
        q (int): qth closest candidate to compute distance for. 
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        candidate_subsets (List[Set[int]]): List of candidate subsets to compute q costs for.

    Returns:
        (np.ndarray): Size n array of distances from voters to candidates.
    """
    q_cst_array = np.zeros((len(candidate_subsets), cst_array.shape[1]))
    for i, subset in enumerate(candidate_subsets):
        q_cst_array[i, :] = q_costs(q, cst_array[list(subset), :])
    return q_cst_array
    
    

def heuristic_group(voter_positions, winner_positions):
    n,_ = voter_positions.shape
    k,_ = winner_positions.shape
    winner_dists = euclidean_cost_array(voter_positions, winner_positions)
    voter_dists = euclidean_cost_array(voter_positions, voter_positions)
    furthest = np.argmax(np.min(winner_dists, axis=0))
    smallest_size = math.ceil(n/k)
    #cohesive_bloc = np.argsort(voter_dists[furthest, :]/np.min(winner_dists, axis=0))[: smallest_size]
    cohesive_bloc = np.argsort(voter_dists[furthest, :])[: smallest_size]
    return cohesive_bloc