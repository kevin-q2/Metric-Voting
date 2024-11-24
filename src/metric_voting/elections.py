# Peter Note IMPORTANT: All of these methods need to be tested in some way. If
# you are not going to use VoteKit which has tests, you need to make sure that
# these methods are spitting out the correct results. If you would like help
# making the tests, you can refer to the ones that are in VoteKit and just copy
# the appropriate ones over needed to test these methods.


import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, Tuple, Any, List, Optional, Union
import pulp
from .utils import approve_profile, remove_candidates, borda_matrix


def SNTV(profile: NDArray, k: int) -> NDArray:
    """
    Elect k candidates with the largest plurality scores.

    Args:
        profile (np.ndarray): (candidates x voters) Preference Profile.
        k (int): Number of candidates to elect

    Returns:
        elected (np.ndarray): Winning candidates
    """
    if not approve_profile(profile):
        raise ValueError("Profile not in correct form.")
    
    first_choice_votes = profile[0, :]
    cands, counts = np.unique(first_choice_votes, return_counts=True)
    
    # break ties randomly with noise
    counts = counts.astype(float)
    counts += np.random.uniform(0, 1, len(counts))
    
    # NOTE: Should this elect random candidates if it can't find k winners??
    elected = cands[np.argsort(counts)[::-1][: min(k, len(cands))]]
    return elected


def Bloc(profile : NDArray, k : int) -> NDArray:
    """
    Elect k candidates with the largest k-approval scores.

    Args:
        profile (np.ndarray): (candidates x voters) Preference Profile.
        k (int): Number of candidates to elect

    Returns:
        elected (np.ndarray): Winning candidates
    """
    if not approve_profile(profile):
        raise ValueError("Profile not in correct form.")
    
    first_choice_votes = profile[:k, :]
    cands, counts = np.unique(first_choice_votes, return_counts=True)
    
    # break ties randomly with noise
    counts = counts.astype(float)
    counts += np.random.uniform(0, 1, len(counts))
    
    # NOTE: Should this elect random candidates if it can't find k winners??
    elected = cands[np.argsort(counts)[::-1][: min(k, len(cands))]]
    return elected



class STV:
    """
    Elect k candidates with the Single Transferrable Vote election.
    Uses the droop quota as an election threshold, and breaks ties randomly.
    To transfer votes, one may select either 'fractional', 'weighted-fractional',
    or 'cambridge' transfer types. The 'fractional' type transfers an 
    equivalent fraction of surplus votes back to voters. The 
    'weighted-fractional' type transfers a fraction of surplus votes back
    to voters proportional to their original vote weight. The 'cambridge'
    type transfers whole surplus votes to a random set of voters who voted for the
    elected candidate.

    Args:
        transfer_type (str): Type of vote transfer. 
            Options: 'fractional', 'weighted-fractional', 'cambridge.'

    Attributes:
        transfer_type (str): Type of vote transfer.
        verbose (bool): Print election details.
        m (int): Number of candidates.
        n (int): Number of voters.
        droop (int): Droop quota.
        elected_mask (NDArray): Tracks elected candidates.
        eliminated_mask (NDArray): Tracks eliminated candidates.
        voter_indices (NDArray): Tracks voter's next available choices.
        voter_weights (NDArray): Weights for each voter's votes.
        elected_count (int): Number of candidates currently elected.
        eliminated_count (int): Number of candidates currently eliminated.
    """
    
    def __init__(self, transfer_type : str = 'fractional', verbose : bool = False):
        self.transfer_type = transfer_type
        self.verbose = verbose
        self.m = None
        self.n = None
        self.droop = None
        self.elected_mask = None
        self.eliminated_mask = None
        self.voter_indices = None
        self.voter_weights = None
        self.elected_count = None
        self.eliminated_count = None
        
    
    def __call__(self, profile : NDArray, k : int) -> NDArray: 
        """
        Elect k candidates with the Single Transferrable Vote election.
        
        Args:
            profile (NDArray): (m x n) Preference Profile.
            k (int): Number of candidates to elect
            
        Returns:
            elected (NDArray): Winning candidates
        """
        self.profile = profile
        self.k = k
        self.m, self.n = profile.shape
        if self.m < k:
            raise ValueError("Requested more elected seats than there are candidates!")
        
        # Initialize tracking variables
        self.droop = int((self.n / (self.k + 1))) + 1
        self.elected_mask = np.zeros(self.m)
        self.eliminated_mask = np.zeros(self.m)
        self.voter_indices = np.zeros(self.n, dtype=int)
        self.voter_weights = np.ones(self.n)
        self.elected_count = 0
        self.eliminated_count = 0
        
        # Main election loop
        if self.verbose:
            print('Starting Election: ')
            
        while self.elected_count < self.k and self.eliminated_count < self.m - self.k:
            candidate_scores, candidate_voters, satisfies_quota = self.count()
            
            if self.verbose:
                print("Round: " + str(self.elected_count + self.eliminated_count))
                print("voter weights: " + str(self.voter_weights))
                print("voter indices: " + str(self.voter_indices))
                print("scores: " + str(candidate_scores))
            
            if len(satisfies_quota) > 0:
                self.elect(satisfies_quota)
                elected = satisfies_quota
                for c in elected:
                    self.transfer(candidate_voters[c])
                    
                if self.verbose:
                    print("elected: " + str(elected))
                
            else:
                elim = self.eliminate(candidate_scores)
                self.transfer(candidate_voters[elim])
                
                if self.verbose:
                    print("eliminated: " + str(elim))
            
            if self.verbose:  
                print()
                
        # Final check: Elect any remaining candidates if needed
        if self.elected_count < self.k:
            remaining_candidates = (np.where((self.elected_mask == 0) 
                                             & (self.eliminated_mask == 0))[0])
            self.elected_mask[remaining_candidates] = 1
            self.elected_count += len(remaining_candidates)
            
        return np.where(self.elected_mask == 1)[0]
                    
                
    def count(self) -> Tuple[NDArray, Dict[int, List[int]], NDArray]:
        """
        Count votes in the current round and determine if any 
        candidates satisfy the droop quota.
        
        Returns:
            candidate_scores (NDArray): Length m array containing the 
                number of votes for each candidate.
            candidate_voters (dict): Dictionary with candidate indices as keys
                and lists of their voters as values.
            satisfies_quota (NDArray): Array of candidate indices that
                satisfy the droop quota.
        """
        candidate_scores = np.zeros(self.m)
        candidate_voters = {c: [] for c in range(self.m)}
        for i in range(self.n):
            if self.voter_indices[i] != -1:
                voter_choice = self.profile[self.voter_indices[i], i]
                
                if self.voter_weights[i] > 0:
                    candidate_scores[voter_choice] += self.voter_weights[i]
                    candidate_voters[voter_choice].append(i)

        satisfies_quota = np.where(
        (candidate_scores >= self.droop) & (self.elected_mask != 1) & (self.eliminated_mask != 1)
        )[0]
        
        return candidate_scores, candidate_voters, satisfies_quota
    
    
    def elect(self, satisfies_quota : NDArray):
        """
        Elect all candidates that satisfy the droop quota.
        
        Args:
            satisfies_quota (NDArray): Array of candidate indices that
                satisfy the droop quota.
        """    
        self.elected_mask[satisfies_quota] = 1
        self.elected_count += len(satisfies_quota)
            
            
    def eliminate(self, candidate_scores : NDArray) -> int:
        """
        Eliminate the candidate with the lowest score, breaking 
        ties randomly.
        
        Args:
            candidate_scores (NDArray): Length m array containing the 
                number of votes for each candidate.
                
        Returns:
            eliminated (int): Index of the eliminated candidate.
        """
        
        random_tiebreakers = np.random.rand(self.m)
        structured_array = np.core.records.fromarrays(
            [candidate_scores, random_tiebreakers], names="scores,rand"
        )
        score_sort = np.argsort(structured_array, order=["scores", "rand"])
        
        eliminated = None
        for e in score_sort:
            if (self.elected_mask[e] == 0) and (self.eliminated_mask[e] == 0):
                self.eliminated_mask[e] = 1
                self.eliminated_count += 1
                eliminated = e

                # only eliminate one candidate per round
                break
            
        return eliminated
    
    
    def transfer(self, candidate_voters : List[int]):
        """
        Transfer votes from an elected candidate to the next available
        candidate on each voter's ballot. Unless there the ballot 
        has been exhausted, in which case the voter is removed 
        from any further voting process. 
        
        Args:
            candidate_voters (List[int]): List of voter indices for 
                voters who voted for the candidate to transfer from.
        """
        total_votes = np.sum(self.voter_weights[candidate_voters])
        surplus_votes = total_votes - self.droop
        
        if surplus_votes >= 0:
            if self.transfer_type == 'fractional':
                self.voter_weights[candidate_voters] = (
                    (surplus_votes / total_votes)
                )
                
            elif self.transfer_type == 'weighted-fractional':
                weights_normalized = (
                    self.voter_weights[candidate_voters] / total_votes
                    )
                self.voter_weights[candidate_voters] = (
                    weights_normalized * surplus_votes
                )
                
            elif self.transfer_type == 'cambridge':
                # NOTE: I think this should work if all voter's weight values are always 1.
                quota_voters = np.random.choice(candidate_voters, 
                                                size = self.droop, 
                                                replace = False)
                self.voter_weights[quota_voters] = 0
                
            else:
                raise ValueError("Invalid transfer type.")
            
        
        self.adjust_indices(candidate_voters)
        
        
    def adjust_indices(self, candidate_voters : List[int]):
        """
        Adjust voter indices to the next available candidate on their ballot.
        If a voter's ballot has been exhausted, remove them from any further
        voting process by setting their voter index to -1.
        
        Args:
            candidate_voters (List[int]): List of voter indices for 
                voters who voted for the candidate to transfer from.
        """
        for v in candidate_voters:
            while self.voter_indices[v] != -1 and (
                (self.elected_mask[self.profile[self.voter_indices[v], v]] == 1)
                or (self.eliminated_mask[self.profile[self.voter_indices[v], v]] == 1)
            ):
                self.voter_indices[v] += 1

                # ballot fully exhausted
                if self.voter_indices[v] >= self.m:
                    self.voter_indices[v] = -1
                    break
 


# Peter Note: Add borda_fn like we did for Borda matrix
def Borda(profile, k):
    """
    Elect k candidates with the largest Borda scores.

    Args:
        profile (np.ndarray): (candidates x voters) Preference Profile.
        k (int): Number of candidates to elect

    Returns:
        elected (np.ndarray): Winning candidates
    """
    if not approve_profile(profile):
        raise ValueError("Profile not in correct form.")
    
    m, n = profile.shape
    candidate_scores = np.zeros(m)

    for i in range(n):
        for j in range(m):
            c = profile[j, i]
            candidate_scores[c] += (m - 1) - j  # Should this be m - j??

    elected = np.argsort(candidate_scores)[::-1][:k]
    return elected


def ChamberlinCourant(profile, k):
    """
    Elect k candidates with the Chamberlain Courant Mechanism.
    This function uses an integer linear program to compute an optimal
    assignment of voters to candidates to maximize the assignment scores
    (where assignment scores are calculated with the borda score).

    Args:
        profile (np.ndarray): (candidates x voters) Preference Profile.
        k (int): Number of candidates to elect

    Returns:
        elected (np.ndarray): Winning candidates
    """
    if not approve_profile(profile):
        raise ValueError("Profile not in correct form.")
    
    m, n = profile.shape
    B = borda_matrix(profile).T  # n x m matrix after transpose

    problem = pulp.LpProblem("Chamberlin-Courant", pulp.LpMaximize)

    # Voter assignment variable (voter j gets assigned to candidate i)
    x = pulp.LpVariable.dicts(
        "x", ((i, j) for i in range(n) for j in range(m)), cat="Binary"
    )
    # Candidate 'elected' variable
    y = pulp.LpVariable.dicts("y", range(m), cat="Binary")

    # Objective function:
    problem += pulp.lpSum(B[i, j] * x[i, j] for i in range(n) for j in range(m))

    # Each voter is assigned to exactly one candidate
    for i in range(n):
        problem += pulp.lpSum(x[i, j] for j in range(m)) == 1

    # A voter can only be assigned to a candidate if that candidate is elected
    for i in range(n):
        for j in range(m):
            problem += x[i, j] <= y[j]

    # Elect exactly k candidates
    problem += pulp.lpSum(y[j] for j in range(m)) == k

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    elected = np.array([j for j in range(m) if pulp.value(y[j]) == 1])
    return elected


def Monroe(profile, k):
    """
    Elect k candidates with the Monroe Mechanism.
    This function uses an integer linear program to compute an optimal
    assignment of voters to candidates to maximize the sum of assignment scores
    (where assignment scores are calculated with the borda score).
    With the added constraint that each candidate can only represent
    exactly floor(n/k) or ceiling(n/k) voters.

    Args:
        profile (np.ndarray): (candidates x voters) Preference Profile.
        k (int): Number of candidates to elect

    Returns:
        elected (np.ndarray): Winning candidates
    """
    if not approve_profile(profile):
        raise ValueError("Profile not in correct form.")
    
    m, n = profile.shape
    B = borda_matrix(profile).T  # n x m matrix after transpose

    problem = pulp.LpProblem("Monroe", pulp.LpMaximize)

    # Voter assignment variable
    x = pulp.LpVariable.dicts(
        "x", ((i, j) for i in range(n) for j in range(m)), cat="Binary"
    )
    # Candidate 'elected' variable
    y = pulp.LpVariable.dicts("y", range(m), cat="Binary")

    # Objective function:
    problem += pulp.lpSum(B[i, j] * x[i, j] for i in range(n) for j in range(m))

    # Each voter is assigned to exactly one candidate
    for i in range(n):
        problem += pulp.lpSum(x[i, j] for j in range(m)) == 1

    # A voter can only be assigned to a candidate if that candidate is elected
    for i in range(n):
        for j in range(m):
            problem += x[i, j] <= y[i]

    # Monroe constraint on the size of candidate's voter sets
    for j in range(m):
        problem += pulp.lpSum(x[i, j] for i in range(n)) >= np.floor(n / k) * y[j]
        problem += pulp.lpSum(x[i, j] for i in range(n)) <= np.ceil(n / k) * y[j]

    # Elect exactly k candidates
    problem += pulp.lpSum(y[j] for j in range(m)) == k

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    elected = np.array([j for j in range(m) if pulp.value(y[j]) == 1])
    return elected


def GreedyCC(profile, k):
    """
    Elect k candidates using a greedy approximation to the
    Chamberlain Courant rule. At every iteration, this rule
    selects a candidate to add to a growing winner set by finding
    the candidate which increases the assignment scores the most.

    Args:
        profile (np.ndarray): (candidates x voters) Preference Profile.
        k (int): Number of candidates to elect

    Returns:
        elected (np.ndarray): Winning candidates
    """
    if not approve_profile(profile):
        raise ValueError("Profile not in correct form.")
    
    m, n = profile.shape
    B = borda_matrix(profile)

    is_elected = np.zeros(m, dtype=bool)
    voter_assign_scores = np.zeros(n) - 1

    for _ in range(k):
        max_score = -1
        max_cand = -1
        for i in range(m):
            if not is_elected[i]:
                score_gain = np.sum(np.maximum(voter_assign_scores, B[i, :]))
                if score_gain > max_score:
                    max_score = score_gain
                    max_cand = i

        is_elected[max_cand] = True
        voter_assign_scores = np.maximum(voter_assign_scores, B[max_cand, :])

    return np.where(is_elected)[0]


# Peter Note: Chunk this out so that you can test the random sections
def PluralityVeto(profile, k):
    """
    Elect k candidates with the Plurality Veto mechanism. Counts
    initial plurality scores for every candidate then, in a randomized order,
    lets voters veto candidates until there are only k left.

    Args:
        profile (np.ndarray): (candidates x voters) Preference Profile.
        k (int): Number of candidates to elect

    Returns:
        elected (np.ndarray): Winning candidates
    """
    if not approve_profile(profile):
        raise ValueError("Profile not in correct form.")
    
    m, n = profile.shape
    candidate_scores = np.zeros(m)
    eliminated = np.zeros(m - k) - 1
    eliminated_count = 0

    # Count initial plurality scores
    first_choice_votes = profile[0, :]
    for c in first_choice_votes:
        candidate_scores[c] += 1

    # Find candidates with 0 plurality score
    zero_scores = np.where(candidate_scores == 0)[0]
    if len(zero_scores) > m - k:
        np.random.shuffle(zero_scores)
        zero_scores = zero_scores[: (m - k)]

    # And remove them from the preference profile
    profile = remove_candidates(profile, zero_scores)
    eliminated[: len(zero_scores)] = zero_scores
    eliminated_count += len(zero_scores)

    # Veto in a randomize order
    random_order = list(range(n))
    np.random.shuffle(random_order)
    while eliminated_count < (m - k):
        for i, v in enumerate(random_order):
            least_preferred = profile[-1, v]
            # A veto decrements the candidates score by 1
            candidate_scores[least_preferred] -= 1
            if candidate_scores[least_preferred] <= 0:
                profile = remove_candidates(profile, [least_preferred])
                eliminated[eliminated_count] = least_preferred
                eliminated_count += 1
                random_order = random_order[i + 1 :] + random_order[: i + 1]
                break

    elected = np.array([c for c in range(m) if c not in eliminated])
    return elected


# Peter Note: Add link to paper
def ExpandingApprovals(profile, k):
    """
    Elect k candidates using the expanding approvals rule seen in:
    (Proportional Representation in Metric Spaces and Low-Distortion Committee Selection
    Kalayci, Kempe, Kher 2024). Please refer to their paper for a full description.

    Args:
        profile (np.ndarray): (candidates x voters) Preference Profile.
        k (int): Number of candidates to elect

    Returns:
        elected (np.ndarray): Winning candidates
    """
    if not approve_profile(profile):
        raise ValueError("Profile not in correct form.")
    
    m, n = profile.shape
    droop = np.ceil(n / k)
    uncovered_mask = np.ones(n, dtype=bool)
    elected_mask = np.zeros(m, dtype=bool)
    neighborhood = np.zeros(profile.shape)
    random_order = np.random.permutation(n)

    # Peter Note: Move some of this to another function so we don't have indent of death
    for t in range(m):
        for v in random_order:
            if uncovered_mask[v]:
                c = profile[t, v]
                if not elected_mask[c]:
                    neighborhood[c, v] = 1
                    if np.sum(neighborhood[c, :]) >= droop:
                        elected_mask[c] = True
                        c_voters = np.where(neighborhood[c, :])[0]
                        neighborhood[:, c_voters] = 0
                        uncovered_mask[c_voters] = False

    if np.sum(elected_mask) < k:
        remaining = k - np.sum(elected_mask)
        non_elected = np.where(elected_mask == False)[0]
        new_elects = np.random.choice(non_elected, remaining, replace=False)
        elected_mask[new_elects] = True

    elected = np.where(elected_mask)[0]
    return elected


def SMRD(profile, k):
    """
    Elect k candidates from k randomly chosen 'dictators'.
    From each dictator elect their first non-elected candidate.

    Args:
        profile (np.ndarray): (candidates x voters) Preference Profile.
        k (int): Number of candidates to elect

    Returns:
        elected (np.ndarray): Winning candidates
    """
    if not approve_profile(profile):
        raise ValueError("Profile not in correct form.")
    
    m, n = profile.shape
    if n < k:
        raise ValueError("Assumes n >= k")
    elected = np.zeros(k, dtype=int) - 1
    elected_mask = np.zeros(m, dtype=bool)
    dictators = np.random.choice(range(n), size=k, replace=False)

    for i in range(k):
        dictator = dictators[i]

        # find next available candidate:
        for j in range(m):
            choice = profile[j, dictator]
            if not elected_mask[choice]:
                elected[i] = choice
                elected_mask[choice] = True
                break

    return elected


def OMRD(profile, k):
    """
    Chooses a single random dictator and lets them elect their top k
    preferences.

    Args:
        profile (np.ndarray): (candidates x voters) Preference Profile.
        k (int): Number of candidates to elect

    Returns:
        elected (np.ndarray): Winning candidates
    """
    if not approve_profile(profile):
        raise ValueError("Profile not in correct form.")
    
    m, n = profile.shape
    dictator = np.random.choice(range(n))
    elected = profile[:k, dictator]
    return elected


def DMRD(profile, k, rho=1):
    """
    Elect k candidates with k iterations of Random Dictator.
    At each iteration, randomly choose a voter and elect its first choice.
    Then discounts or reweights the voting power of voters who voted for that candidate by
    a factor of rho.

    Args:
        profile (np.ndarray): (candidates x voters) Preference Profile.
        k (int): Number of candidates to elect
        rho (float): Reweighting factor

    Returns:
        elected (np.ndarray): Winning candidates
    """
    if not approve_profile(profile):
        raise ValueError("Profile not in correct form.")
    
    m, n = profile.shape
    voter_probability = np.ones(n) / n
    voter_indices = np.zeros(n, dtype=int)
    elected = np.zeros(k, dtype=int) - 1
    elected_mask = np.zeros(m, dtype=bool)

    for i in range(k):
        dictator = np.random.choice(range(n), p=voter_probability)

        # find next available candidate:
        winner = -1
        for j in range(m):
            choice = profile[j, dictator]
            if not elected_mask[choice]:
                winner = choice
                elected[i] = choice
                elected_mask[choice] = True
                break

        # Find who voted for the winner
        first_choice_votes = profile[voter_indices, np.arange(n)]
        mask = first_choice_votes == winner

        # Adjusts voter probability for the next round
        voter_probability[mask] *= rho
        voter_probability /= np.sum(voter_probability)

        # Effectively removes winning candidate from profile
        voter_indices[mask] += 1

    return elected


def PRD(profile, k, p=None, q=None):
    """
    Elect k candidates with k iterations of Proportional Random Dictator (PRD).
    At each iteration, with probability p choose a winner with
    the proportional to q's rule. With probability (1-p) instead choose a
    winner with RandomDictator. Remove that candidate from all preference profiles and repeat.

    Args:
        profile (np.ndarray): (candidates x voters) Preference Profile.
        k (int): Number of candidates to elect
        p (float, optional): probability of proportional to q's rule
        q (float, optional): power in proportional to q's

    Returns:
        elected (np.ndarray): Winning candidates
    """
    if not approve_profile(profile):
        raise ValueError("Profile not in correct form.")

    m, n = profile.shape

    # Default setting
    if p is None:
        p = 1 / (m - 1)
    if q is None:
        q = 2

    voter_probability = np.ones(n) / n
    voter_indices = np.zeros(n, dtype=int)
    elected = np.zeros(k, dtype=int) - 1
    for i in range(k):
        first_choice_votes = profile[voter_indices, np.arange(n)]

        coin_flip = np.random.uniform()
        if coin_flip <= p:
            cands, counts = np.unique(first_choice_votes, return_counts=True)
            prob = np.power(counts * 1.0, q)
            prob /= np.sum(prob)
            winner = np.random.choice(cands, p=prob)
        else:
            # winner = np.random.choice(first_choice_votes)
            dictator = np.random.choice(range(n), p=voter_probability)
            winner = profile[voter_indices[dictator], dictator]

        elected[i] = winner

        # removes winning candidate from profile
        mask = first_choice_votes == winner

        # Adjusts voter probability for the next round
        # (If we did random dictator)
        voter_probability[mask] *= 0.8
        voter_probability /= np.sum(voter_probability)

        # Effectively removes winning candidate from profile
        voter_indices[mask] += 1

    return elected
