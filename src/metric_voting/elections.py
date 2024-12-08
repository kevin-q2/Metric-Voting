import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, Tuple, Any, List, Optional, Union
import pulp
from .utils import remove_candidates, borda_matrix


####################################################################################################


class Election:
    """
    Base class for election types.
    """
    
    def __init__(self):
        pass
    
    def _is_complete_ranking(self, ranking: NDArray) -> bool:
        """
        Checks if the input ranking is a complete ranking of the same m candidates.

        Args:
            ranking (np.ndarray): (m) Array of candidate indices.

        Returns:
            (bool): True if the ranking is complete, False otherwise.
        """
        return np.all(np.isin(np.arange(len(ranking)), np.unique(ranking)))


    def _approve_profile(self, profile: NDArray) -> bool:
        """
        Checks if the input profile is an ranked preference profile in 
        correct form. Specifically, for our simplified models, this means 
        a complete ranking of the same m candidates for each voter. 

        Args:
            profile (np.ndarray): (candidates x voters) Preference profile matrix.

        Returns:
            (bool): True if the profile is approved, False otherwise.
        """
        return np.all(np.apply_along_axis(self._is_complete_ranking, axis = 0, arr = profile))
            
    
    def elect(profile: NDArray, k: int) -> NDArray:
        """
        Elect k candidates using an input preference profile.
        
        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        
        pass
    
    
####################################################################################################


class SNTV(Election):
    """
    Single Non-Transferable Vote (Plurality) election.
    """
    def __init__(self):
        pass 
    
    
    def elect(self, profile: NDArray, k: int) -> NDArray:
        """
        Elect k candidates with the largest plurality scores.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        if not self._approve_profile(profile):
            raise ValueError("Profile not in correct form.")
        
        first_choice_votes = profile[0, :]
        cands, counts = np.unique(first_choice_votes, return_counts=True)
        
        # break ties randomly with noise
        counts = counts.astype(float)
        counts += np.random.uniform(0, 1, len(counts))
        
        # NOTE: Should this elect random candidates if it can't find k winners??
        elected = cands[np.argsort(counts)[::-1][: min(k, len(cands))]]
        return elected


####################################################################################################


class CommitteeScoring(Election):
    """
    General parent class for electing k candidates via simple Committee Scoring Mechanisms.
    This class uses a scoring scheme to give scores to candidates based upon
    their position in the preference profile. This is a function 
    which takes as input the number of candidates m, the number of winners k, 
    and the position i in {1,...,m} of the candidate in a given voter's ranking. 
    
    Scores for a candidate are summed over all voters, and the k candidates with the
    largest scores are elected.
    
    Args:
        scoring_scheme (Callable[[int, int, int], float]): Scheme for candidate scoring. 
    """
    def __init__(self,scoring_scheme: Callable[[int, int, int], float]):        
        self.scoring_scheme = scoring_scheme
        
    def elect(self, profile : NDArray, k : int) -> NDArray: 
        """
        Elect k candidates with the largest Borda scores.
        
        Args:
            profile (NDArray): (m x n) Preference Profile.
            k (int): Number of candidates to elect
            
        Returns:
            elected (NDArray): Winning candidates
        """
        
        if not self._approve_profile(profile):
            raise ValueError("Profile not in correct form.")
        
        m, n = profile.shape
        candidate_scores = np.zeros(m)

        for i in range(n):
            for j in range(m):
                c = profile[j, i]
                candidate_scores[c] += self.scoring_scheme(m, k, j + 1)
                
        # break ties randomly with noise
        candidate_scores = candidate_scores.astype(float)
        candidate_scores += np.random.uniform(0, 1, len(candidate_scores))

        elected = np.argsort(candidate_scores)[::-1][:k]
        return elected
    
    
####################################################################################################


class Bloc(CommitteeScoring):
    """
    Bloc approval election method. This is a committee scoring method
    which uses the scoring scheme f(m, k, i) = 1 for i <= k and 0 otherwise.
    """
    def __init__(self):
        scoring_scheme = lambda x, y, z: 1 if z <= y else 0
        super().__init__(scoring_scheme)
    

####################################################################################################


class Borda(CommitteeScoring):
    """
    Elect k candidates with the Borda Scoring Mechanism. This is a committee scoring method 
    which uses the scoring scheme f(m, k, i) = m - i.
    """
    def __init__(self):        
        scoring_scheme = lambda x, y, z: x - z
        super().__init__(scoring_scheme)
    
    
####################################################################################################


class STV(Election):
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
        
    
    def elect(self, profile : NDArray, k : int) -> NDArray: 
        """
        Elect k candidates with the Single Transferrable Vote election.
        
        Args:
            profile (NDArray): (m x n) Preference Profile.
            k (int): Number of candidates to elect
            
        Returns:
            elected (NDArray): Winning candidates
        """
        if not self._approve_profile(profile):
            raise ValueError("Profile not in correct form.")
        
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
                self.add_candidates(satisfies_quota)
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
    
    
    def add_candidates(self, satisfies_quota : NDArray):
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
                    (surplus_votes / len(candidate_voters))
                )
                
            elif self.transfer_type == 'weighted-fractional':
                weights_normalized = (
                    self.voter_weights[candidate_voters] / total_votes
                    )
                self.voter_weights[candidate_voters] = (
                    surplus_votes * weights_normalized
                )
                
            elif self.transfer_type == 'cambridge':
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
 

####################################################################################################


class ChamberlinCourant(Election):
    """
    Elect k candidates with the Chamberlain Courant Mechanism.
    This function uses an integer linear program to compute an optimal
    assignment of voters to candidates to maximize the assignment scores
    (where assignment scores are calculated with the borda score).
    
    NOTE: As far as I can tell, these solvers output deterministic answers,
    even in cases where there are multiple optimal solutions...
    
    Args:
        solver (str, optional): Solver for the integer linear program. These are taken 
            from PuLP's available solvers, for more information please see 
            (https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html).
            
            Some options 'PULP_CBC_CMD' and 'GUROBI_CMD' (requires licesnse).
            Defaults to 'PULP_CBC_CMD', which uses PuLPs default coin and branch bound solver.
            
    Attributes:
        objective (float): Objective value of the last solved problem.
    """
    def __init__(self, solver : str = 'PULP_CBC_CMD'):
        self.solver = pulp.getSolver(solver, msg = False)
        self.objective = None
        
    
    def elect(self, profile : NDArray, k : int) -> NDArray: 
        """
        Elect k candidates with the Chamberlain Courant Mechanism.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        if not self._approve_profile(profile):
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

        problem.solve(self.solver)
        elected = np.array([j for j in range(m) if pulp.value(y[j]) == 1])
        self.objective = pulp.value(problem.objective)
        return elected
    
    
####################################################################################################


class Monroe(Election):
    """
    Elect k candidates with the Monroe Mechanism.
    This function uses an integer linear program to compute an optimal
    assignment of voters to candidates to maximize the sum of assignment scores
    (where assignment scores are calculated with the borda score).
    With the added constraint that each candidate can only represent
    exactly floor(n/k) or ceiling(n/k) voters.
    
    NOTE: As far as I can tell, these solvers output deterministic answers,
    even in cases where there are multiple optimal solutions...
    
    Args:
        solver (str, optional): Solver for the integer linear program. These are taken 
            from PuLP's available solvers, for more information please see 
            (https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html).
            
            Some options 'PULP_CBC_CMD' and 'GUROBI_CMD' (requires licesnse).
            Defaults to 'PULP_CBC_CMD', which uses PuLPs default coin and branch bound solver.
    
    Attributes:
        objective (float): Objective value of the last solved problem.
    """
    def __init__(self, solver : str = 'PULP_CBC_CMD'):
        self.solver = pulp.getSolver(solver, msg = False)
    
    
    def elect(self, profile : NDArray, k : int) -> NDArray: 
        """

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        if not self._approve_profile(profile):
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
                problem += x[i, j] <= y[j]

        # Monroe constraint on the size of candidate's voter sets
        # This is the only difference from chamberlin Courant
        for j in range(m):
            problem += pulp.lpSum(x[i, j] for i in range(n)) >= np.floor(n / k) * y[j]
            problem += pulp.lpSum(x[i, j] for i in range(n)) <= np.ceil(n / k) * y[j]

        # Elect exactly k candidates
        problem += pulp.lpSum(y[j] for j in range(m)) == k

        problem.solve(self.solver)
        elected = np.array([j for j in range(m) if pulp.value(y[j]) == 1])
        self.objective = pulp.value(problem.objective)
        return elected
    
    
####################################################################################################


class GreedyCC(Election):
    """
    Elect k candidates using a greedy approximation to the
    Chamberlain Courant rule. At every iteration, this rule
    selects a candidate to add to a growing winner set by greedily selecting
    the candidate which gives the best increase to the current assignment scores.
    
    For more information, please see the paper:
    "Budgeted Social Choice: From Consensus to Personalized Decision Making" 
    - Lu and Boutilier (2011)
    (https://www.cs.toronto.edu/~tl/papers/LuBoutilier_budgeted_IJCAI11.pdf)
    
    
    Attributes:
        objective (float): Objective value of the last solved problem.
    """
    def __init__(self):
        self.objective = None
    
    
    def elect(self, profile : NDArray, k : int) -> NDArray:
        """
        Elect k candidates using a greedy approximation to the
        Chamberlain Courant rule.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        if not self._approve_profile(profile):
            raise ValueError("Profile not in correct form.")
        
        m, n = profile.shape
        B = borda_matrix(profile)

        is_elected = np.zeros(m, dtype=bool)
        voter_assign_scores = np.zeros(n) - 1

        for _ in range(k):
            scores = np.zeros(m)
            for i in range(m):
                if not is_elected[i]:
                    score_gain = np.sum(np.maximum(voter_assign_scores, B[i, :]))
                    scores[i] = score_gain

            # Break ties randomly
            scores += np.random.uniform(0, 1, len(scores))
            max_cand = np.argmax(scores)
            is_elected[max_cand] = True
            voter_assign_scores = np.maximum(voter_assign_scores, B[max_cand, :])

        self.objective = np.sum(voter_assign_scores)
        return np.where(is_elected)[0]


####################################################################################################


class PluralityVeto(Election):
    """
    Elect k candidates with the Plurality Veto mechanism. Counts
    initial plurality scores for every candidate then, in a randomized order,
    lets voters veto candidates until there are only k left. 
    
    For more information please see the paper:
    "Plurality Veto: A Simple Voting Rule Achieving Optimal Metric Distortion"
    - Kizilkaya, Kempe (2023)
    (https://arxiv.org/abs/2206.07098)
    
    NOTE: That this is an extension for what is really designed to be a single winner 
    voting rule. 
    """
    def __init__(self):
        self.profile = None
        self.m, self.n = None, None
        self.k = None
        self.candidate_scores = None
        self.eliminated = None
        self.last_place_indices = None
    
    
    def update_least_preffered(self):
        """
        Updates the last place indices of voters in the preference profile.
        """
        for i in range(self.n):
            while self.eliminated[self.profile[self.last_place_indices[i], i]] == 1:
                self.last_place_indices[i] -= 1
                
    
    def initialize(self):
        """
        Counts first place votes and eliminates any candidates with 0.
        """
        # Count initial plurality scores
        first_choice_votes = self.profile[0, :]
        for c in first_choice_votes:
            self.candidate_scores[c] += 1

        # Find candidates with 0 plurality score
        zero_scores = np.where(self.candidate_scores == 0)[0]
        if len(zero_scores) > self.m - self.k:
            np.random.shuffle(zero_scores)
            zero_scores = zero_scores[: (self.m - self.k)]

        # And remove them from the election
        self.eliminated[zero_scores] = 1
        self.update_least_preffered()
        
        
    def elect(self, profile : NDArray, k : int) -> NDArray:
        """
        Elect k candidates with the multiwinner Plurality Veto mechanism.
        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        
        if not self._approve_profile(profile):
            raise ValueError("Profile not in correct form.")
        
        self.profile = profile
        self.k = k
        self.m, self.n = profile.shape
        self.candidate_scores = np.zeros(self.m)
        self.eliminated = np.zeros(self.m)
        self.last_place_indices = np.zeros(self.n, dtype=int) + self.m - 1 # Last place index
        
        self.initialize()
        eliminated_count = len(np.where(self.eliminated == 1)[0])

        # Veto in a randomize order
        random_order = list(range(self.n))
        np.random.shuffle(random_order)
        while eliminated_count < (self.m - self.k):
            for i, v in enumerate(random_order):
                least_preferred = profile[self.last_place_indices[v], v]
                # A veto decrements the candidates score by 1
                self.candidate_scores[least_preferred] -= 1
                if self.candidate_scores[least_preferred] <= 0:
                    self.eliminated[least_preferred] = 1
                    self.update_least_preffered()
                    eliminated_count += 1
                    random_order = random_order[i + 1 :] + random_order[: i + 1]
                    break

        elected = np.where(self.eliminated == 0)[0]
        return elected
    

####################################################################################################


class MultiPluralityVeto(Election):
    """
    Elect k candidates with the Multiwinner Plurality Veto mechanism. Each voter 
    ranks potential winner sets, which the mechanism then treats like individual 
    candidates. The mechanism then elects a winner set by applying the 
    single winner Plurality Veto rule to this ranking.
    
    For more information please see the paper:
    "Plurality Veto: A Simple Voting Rule Achieving Optimal Metric Distortion"
    - Kizilkaya, Kempe (2023)
    (https://arxiv.org/abs/2206.07098)
    """
    
    def __init__(self):
        pass
    
    def elect(profile : NDArray, k : int) -> NDArray:
        pass


####################################################################################################


class ExpandingApprovals(Election):
    """
    Elect k candidates using the expanding approvals rule.
    Candidates are elected through a sequential approval process. At each step i, 
    voters approve their ith ranked candidate in a random order. Once candidates 
    have passed an approval threshold, they are elected. 
    
    For more information please see the paper:
    "Proportional Representation in Metric Spaces and Low-Distortion Committee Selection"
    -Kalayci, Kempe, Kher (2024)
    (https://arxiv.org/abs/2312.10369)
    """
    def __init__(self):
        self.quota = None
        self.uncovered_mask = None
        self.elected_mask = None
        self.neighborhood = None
        self.random_order = None
    
    
    def candidate_check_elect(self, c : int):
        """
        Check if a candidate has passed the quota and elect them if they have.
        
        Args:
            c (int): Candidate index.
        """
        if np.sum(self.neighborhood[c, :]) >= self.quota:
            self.elected_mask[c] = True
            c_voters = np.where(self.neighborhood[c, :])[0]
            self.neighborhood[:, c_voters] = 0
            self.uncovered_mask[c_voters] = False
        
    
    def voter_approve(self, v : int, t : int):
        """
        Approve the voter's t-th ranked candidate if they have not 
        yet been elected, and check if they have reached the quota.
        
        Args:
            voter (int): Voter index.
            t (int): Rank of the voter's candidate.
        """
        if self.uncovered_mask[v]:
            c = self.profile[t, v]
            if not self.elected_mask[c]:
                self.neighborhood[c, v] = 1
                self.candidate_check_elect(c)
        
    
    def approval_round(self, t : int):
        """
        Conduct an approval round of the expanding approvals rule.
        
        Args:
            t (int): Round number.
        """
        for v in self.random_order:
            self.voter_approve(v, t)
        
    
    def elect(self, profile : NDArray, k : int) -> NDArray:
        """
        Elect k candidates using the expanding approvals rule. 

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        if not self._approve_profile(profile):
            raise ValueError("Profile not in correct form.")
        
        self.profile = profile
        m, n = profile.shape
        self.quota = np.ceil(n / k)
        self.uncovered_mask = np.ones(n, dtype=bool)
        self.elected_mask = np.zeros(m, dtype=bool)
        self.neighborhood = np.zeros(profile.shape)
        self.random_order = np.random.permutation(n)

        # Main election loop
        for t in range(m):
            self.approval_round(t)
            
        # Elect remaining candidates if needed
        if np.sum(self.elected_mask) < k:
            remaining = k - np.sum(self.elected_mask)
            non_elected = np.where(self.elected_mask == False)[0]
            new_elects = np.random.choice(non_elected, remaining, replace=False)
            self.elected_mask[new_elects] = True

        elected = np.where(self.elected_mask)[0]
        return elected


####################################################################################################


class SMRD(Election):
    """
    Sequential Multiwinner Random Dictator (SMRD) election method.
    Elect k candidates from k randomly chosen 'dictators'.
    From each dictator elect their first non-elected candidate.
    """
    def __init__(self):
        pass
    
    def elect(self, profile : NDArray, k : int) -> NDArray:
        """
        Elect k candidates with the SMRD method.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        if not self._approve_profile(profile):
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


####################################################################################################

class OMRD(Election):
    """
    One-Shot Multiwinner Random Dictator (OMRD) election method.
    Chooses a single random dictator and lets them elect their top k
    preferences.
    """
    def __init__(self):
        pass 
    
    def elect(self, profile : NDArray, k : int) -> NDArray:
        """
        Elect k winners with the OMRD method.
        
        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        if not self._approve_profile(profile):
            raise ValueError("Profile not in correct form.")
        
        m, n = profile.shape
        dictator = np.random.choice(range(n))
        elected = profile[:k, dictator]
        return elected


####################################################################################################


class DMRD(Election):
    """
    Discounted Multiwinner Random Dictator (DMRD) election method.
    Elect k candidates with k iterations of Random Dictator.
    At each iteration, randomly choose a voter and elect its first choice.
    Then discount or reweight the voting power of voters who voted for that candidate by
    a factor of rho.

    Args:
        rho (float, optional): Reweighting factor. Default is 1.
    """
    def __init__(self, rho : int = 1):
        self.rho = rho
        
    def elect(self, profile : NDArray, k : int) -> NDArray:
        """
        Elect k candidates with the DMRD method.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        if not self._approve_profile(profile):
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
            voter_probability[mask] *= self.rho
            voter_probability /= np.sum(voter_probability)

            # Effectively removes winning candidate from profile
            voter_indices[mask] += 1

        return elected


####################################################################################################

'''
# Not Currently in Use:
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
'''
