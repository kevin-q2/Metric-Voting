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


def cost_array_to_ranking(cst_array: NDArray) -> NDArray:
    """
    Given a cost array, returns a ranking of candidates for each voter.
    Specifically, for an (m x n) cost array, the output is an (m x n) ranking matrix
    where entry [i,j] stores the rank of candidate i for voter j.

    Args:
        cst_array (np.ndarray): (m x n) Cost array.

    Returns:
        (np.ndarray): Computed (m x n) ranking matrix.
    """
    return np.argsort(cst_array, axis=0)


def tiebreak(scores : NDArray, proxy : NDArray = None) -> NDArray:
    """
    Breaks ties in a length m array of scores by:
    1) (IF given) Comparing values in a same-sized proxy array.
    2) Otherwise breaking ties randomly. 
    
    Args:
        scores (np.ndarray): Length m array of scores.
        proxy (np.ndarray, optional): Length m array of proxy values to use for tiebreaking.
            Defaults to None.
            
    Returns:
        argsort (np.ndarray): Length m argsort of scores with ties broken.
    """
    m = len(scores)
    random_tiebreakers = np.random.rand(m)
    if proxy is not None:
        return np.lexsort((random_tiebreakers, proxy, scores))
    else:
        return np.lexsort((random_tiebreakers, scores))


def borda_matrix(
    profile: NDArray, scoring_scheme: Callable[[int, int], float] = lambda x, y: x - y
) -> NDArray:
    """
    Computes a borda matrix given an input preference profile.
    Specifically, for an (m x n) profile the borda matrix at
    entry [i,j] records the borda score of candidate i for voter j.

    Args:
        profile (np.ndarray): (candidates x voters) Preference profile matrix.
        
        scoring_scheme (callable, optional): Scoring scheme to use for borda scoring.
            Defaults to the standard borda scoring scheme which is equivalent to 
            a function f(m, i) = m - i where m is the total number of candidates 
            and i is a voter's ranking in [1,...,m] of the candidate. 

    Returns:
        (np.ndarray): Computed (m x n) borda matrix.
    """
    m, n = profile.shape
    B = np.zeros((m, n))
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            B[profile[i, j], j] = scoring_scheme(m, i + 1)

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



def random_voter_bloc(n : int, k : int, t : int, weights : NDArray) -> NDArray:
    """
    Generates a random bloc of voters given size constraints and probabilites for 
    selection. The bloc size is determined by the parameters n, k, and t. We say that 
    for an election with n voters and k winners, a bloc that 'deserves' t <= k representatives
    has size [n*t/k, n*(t+1)/k - 1]. The bloc generated by uniformly choosing a size 
    within this range, and then choosing that number of voters randomly from 
    the set of n voters with probabilities given by the weights array.
    
    Args:
        n (int): Number of voters.
        k (int): Number of winners.
        t (int): Number of representatives the bloc deserves.
        weights (np.ndarray): Length n array of probabilities for each voter to be selected.
    
    Returns:
        random_bloc (np.ndarray): Randomly selected bloc of voters.
    """
    if t != k:
        min_size = math.ceil(n * t / k)
        max_size = min(math.ceil(n * (t + 1) / k) - 1, n)
        bloc_size = np.random.randint(min_size, max_size)
        random_bloc = np.random.choice(n, bloc_size, replace=False, p = weights)
    else:
        random_bloc = np.arange(n)
    return random_bloc


def geq_with_tol(a, b, tol = 1e-12):
    """
    Checks if a is greater than or equal to b with a given tolerance.
    
    Args:
        a (float): First number.
        b (float): Second number.
        tol (float, optional): Tolerance. Defaults to 1e-12.
    
    Returns:
        (bool): True if a >= b, False otherwise.
    """
    return a >= b-tol