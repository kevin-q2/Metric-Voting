import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, Tuple, Any, List, Optional, Union
import pulp
from .utils import tiebreak, remove_candidates, borda_matrix, geq_with_tol


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


    def _approve_profile(self, profile: NDArray, k : int) -> bool:
        """
        Checks if the input profile is an ranked preference profile in 
        correct form. Specifically, for our simplified models, this means 
        a complete ranking of the same m candidates for each voter. 

        Args:
            profile (np.ndarray): (candidates x voters) Preference profile matrix.
            
            k (int): Number of candidates to elect.

        Returns:
            (bool): True if the profile is approved, False otherwise.
        """
        m,n = profile.shape
        if n < k:
            raise ValueError("Requested more elected seats than there are voters!")
        if m < k:
            raise ValueError("Requested more elected seats than there are candidates!")
        
        if not np.all(np.apply_along_axis(self._is_complete_ranking, axis = 0, arr = profile)):
            raise ValueError("Profile not in correct form.")
            
    
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
    
    NOTE: Should this elect random candidates if it can't find k winners??
    """
    def __init__(self):
        pass 
    
    
    def elect(self, profile: NDArray, k: int) -> NDArray:
        """
        Elect k candidates with the largest plurality scores. If k candidates are not 
        named among the first choices, the election moves on to consider the second choices, 
        and so on until k candidates are elected.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """        
        m,_ = profile.shape
        self._approve_profile(profile, k)   
        elected_mask = np.zeros(m, dtype = bool)
        elected_count = 0
        counts = np.zeros(m)
        
        current_rank = 0
        while elected_count < k:
            votes = profile[current_rank, :]
            for c in votes:
                counts[c] += 1
            
            ranking = tiebreak(counts)[::-1]
            for c in ranking:
                if (elected_count < k) and (not elected_mask[c]) and (counts[c] > 0):
                    elected_mask[c] = True
                    elected_count += 1
            
            current_rank += 1
            
        return np.where(elected_mask)[0]
        


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
        Elect k candidates with the largest scores.
        
        Args:
            profile (NDArray): (m x n) Preference Profile.
            k (int): Number of candidates to elect
            
        Returns:
            elected (NDArray): Winning candidates
        """
        
        self._approve_profile(profile, k)
        m, n = profile.shape
        candidate_scores = np.zeros(m)

        for i in range(n):
            for j in range(m):
                c = profile[j, i]
                candidate_scores[c] += self.scoring_scheme(m, k, j + 1)

        ranking = tiebreak(candidate_scores)[::-1]
        elected = ranking[:k]
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
            Options: 'fractional', 'weighted-fractional', 'cambridge.' Defaults to 
            'weighted-fractional'.
        tiebreak_type (str) : Method for breaking ties. Options: 'fpv_random', 'random'.
            Defaults to 'fpv_random', which breaks ties by comparing first round scores, 
            and if that fails, breaks ties randomly. Otherwise, the 'random' option breaks
            ties randomly.

    Attributes:
        transfer_type (str): Type of vote transfer.
        tiebreak_type (str): Method for breaking ties.
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
        first_round_scores (NDArray): Scores for the first round of candidates. 
    """
    
    def __init__(
        self,
        transfer_type : str = 'weighted-fractional',
        tiebreak_type : str = 'fpv_random',
        verbose : bool = False,
    ):
        if transfer_type not in ['fractional', 'weighted-fractional', 'cambridge']:
            raise ValueError("Invalid transfer type.")
        self.transfer_type = transfer_type
        
        self.tiebreak_type = tiebreak_type
        if tiebreak_type not in ['fpv_random', 'random']:
            raise ValueError("Invalid tiebreak type.")
        
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
        self.first_round_scores = None
        
    
    def elect(self, profile : NDArray, k : int) -> NDArray: 
        """
        Elect k candidates with the Single Transferrable Vote election.
        
        Args:
            profile (NDArray): (m x n) Preference Profile.
            k (int): Number of candidates to elect
            
        Returns:
            elected (NDArray): Winning candidates
        """
        self._approve_profile(profile, k)
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
        self.voter_weights = np.ones(self.n, dtype=np.float64)
        self.elected_count = 0
        self.eliminated_count = 0
        
        # Main election loop
        if self.verbose:
            print('Starting Election: ')
            
        while self.elected_count < self.k and self.eliminated_count < self.m - self.k:
            candidate_scores, candidate_voters, satisfies_quota = self.count()
            
            if self.elected_count + self.eliminated_count == 0:
                self.first_round_scores = candidate_scores
            
            if self.verbose:
                print("Round: " + str(self.elected_count + self.eliminated_count))
                print("voter weights: " + str(self.voter_weights))
                print("voter indices: " + str(self.voter_indices))
                print("scores: " + str(candidate_scores))
                print("sum of scores: " + str(np.sum(candidate_scores)))
                print("sum of weights: " + str(np.sum(self.voter_weights)))
            
            if len(satisfies_quota) > 0:
                self.add_candidates(satisfies_quota)
                elected = satisfies_quota
                for c in elected:
                    self.transfer(candidate_scores[c], candidate_voters[c])
                    
                if self.verbose:
                    print("elected: " + str(elected))
                
            else:
                elim = self.eliminate(candidate_scores)
                self.transfer(candidate_scores[elim], candidate_voters[elim])
                
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
        candidate_scores = np.zeros(self.m, dtype = np.float64)
        candidate_voters = {c: [] for c in range(self.m)}
        for i in range(self.n):
            if self.voter_indices[i] != -1:
                voter_choice = self.profile[self.voter_indices[i], i]
                
                if self.voter_weights[i] > 0:
                    #candidate_scores[voter_choice] += self.voter_weights[i]
                    candidate_voters[voter_choice].append(i)

        for c, voters in candidate_voters.items():
            candidate_scores[c] = np.sum(self.voter_weights[voters])
            
        satisfies_quota = np.where(
        (geq_with_tol(candidate_scores, self.droop, tol = 1e-10)) & 
        (self.elected_mask != 1) & 
        (self.eliminated_mask != 1)
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
        if self.tiebreak_type == 'fpv_random':
            score_sort = tiebreak(candidate_scores, self.first_round_scores)
        else:
            score_sort = tiebreak(candidate_scores)
            
        
        eliminated = None
        for e in score_sort:
            if (self.elected_mask[e] == 0) and (self.eliminated_mask[e] == 0):
                self.eliminated_mask[e] = 1
                self.eliminated_count += 1
                eliminated = e

                # only eliminate one candidate per round
                break
            
        return eliminated
    
    
    def transfer(self, total_votes: float, candidate_voters : List[int]):
        """
        Transfer votes from an elected candidate to the next available
        candidate on each voter's ballot. Unless there the ballot 
        has been exhausted, in which case the voter is removed 
        from any further voting process. 
        
        Args:
            votes (float): Number of votes to transfer.
            candidate_voters (List[int]): List of voter indices for 
                voters who voted for the candidate to transfer from.
        """
        surplus_votes = total_votes - self.droop
        
        if self.verbose:
            print("surplus: " + str(surplus_votes))
        
        if geq_with_tol(surplus_votes, 0, tol = 1e-10):
            if self.transfer_type == 'fractional':
                self.voter_weights[candidate_voters] = np.maximum(
                    (surplus_votes / len(candidate_voters)),
                    0.0
                )
                
            elif self.transfer_type == 'weighted-fractional':
                weights_normalized = (
                    self.voter_weights[candidate_voters] / total_votes
                    )
                self.voter_weights[candidate_voters] = np.maximum(
                    surplus_votes * weights_normalized,
                    0.0
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
    even in cases where there are multiple optimal solutions...Not sure how to handle this.
    
    Args:
        solver (str, optional): Solver for the integer linear program. These are taken 
            from PuLP's available solvers, for more information please see 
            (https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html).
            
            Some options 'PULP_CBC_CMD' and 'GUROBI_CMD' (requires licesnse).
            Defaults to 'PULP_CBC_CMD', which uses PuLPs default coin and branch bound solver.
            
        log_path (str, optional): Path to log file for solver output. Defaults to 'cc.log'.
            
    Attributes:
        objective (float): Objective value of the last solved problem.
    """
    def __init__(self, solver : str = 'PULP_CBC_CMD', log_path : str = None):
        self.log_path = log_path
        self.solver = pulp.getSolver(
            solver,
            msg = False,
            logPath = log_path
        )
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
        # Creates a new log file for each election
        if self.log_path is not None:
            with open(self.log_path, 'w') as f:
                f.write('')
                
        self._approve_profile(profile, k)       
        m, n = profile.shape
        B = borda_matrix(profile, k).T  # n x m matrix after transpose

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
    even in cases where there are multiple optimal solutions...Not sure how to handle this.
    
    Args:
        solver (str, optional): Solver for the integer linear program. These are taken 
            from PuLP's available solvers, for more information please see 
            (https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html).
            
            Some options 'PULP_CBC_CMD' and 'GUROBI_CMD' (requires licesnse).
            Defaults to 'PULP_CBC_CMD', which uses PuLPs default coin and branch bound solver.
            
        log_path (str, optional): Path to log file for solver output. Defaults to 'monroe.log'.
    
    Attributes:
        objective (float): Objective value of the last solved problem.
    """
    def __init__(self, solver : str = 'PULP_CBC_CMD', log_path : str = None):
        self.log_path = log_path
        self.solver = pulp.getSolver(
            solver,
            msg = False,
            logPath = log_path  
        )
    
    
    def elect(self, profile : NDArray, k : int) -> NDArray: 
        """

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        # Creates a new log file for each election
        if self.log_path is not None:
            with open(self.log_path, 'w') as f:
                f.write('')
                
        self._approve_profile(profile, k)
        m, n = profile.shape
        B = borda_matrix(profile, k).T  # n x m matrix after transpose

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


class PAV(Election):
    """
    Elect k candidates with the harmonic comittee scoring mechanism..
    This function uses an integer linear program to find a winner set which
    maximizes the sum of harmonic adjusted committee scores.
    With an approval scoring scheme, this is equivalent to the PAV rule.
    With a borda scoring scheme, this is equivalent to the Harmonic Borda rule.
    
    For more information, please see:
    "What Do Multiwinner Voting Rules Do? An Experiment Over the Two-Dimensional Euclidean Domain"
    Elkind et al (2019)
    (https://arxiv.org/abs/1901.09217)
    
    NOTE: As far as I can tell, these solvers output deterministic answers,
    even in cases where there are multiple optimal solutions...Not sure how to handle this.
    
    Args:
        solver (str, optional): Solver for the integer linear program. These are taken 
            from PuLP's available solvers, for more information please see 
            (https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html).
            
            Some options 'PULP_CBC_CMD' and 'GUROBI_CMD' (requires licesnse).
            Defaults to 'PULP_CBC_CMD', which uses PuLPs default coin and branch bound solver.
            
        log_path (str, optional): Path to log file for solver output. Defaults to None.
    
    Attributes:
        objective (float): Objective value of the last solved problem.
    """
    def __init__(
        self,
        scoring_scheme : Callable = lambda x, y, z: 1 if z <= y else 0,
        solver : str = 'PULP_CBC_CMD',
        log_path : str = None
    ):
        self.scoring_scheme = scoring_scheme
        self.log_path = log_path
        self.solver = pulp.getSolver(
            solver,
            msg = False,
            logPath = log_path
        )
    
    
    def elect(self, profile : NDArray, k : int) -> NDArray: 
        """

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        # Creates a new log file for each election
        if self.log_path is not None:
            with open(self.log_path, 'w') as f:
                f.write('')
                
        self._approve_profile(profile, k)
        m, n = profile.shape
        B = borda_matrix(profile, k, self.scoring_scheme).T  # n x m matrix after transpose

        problem = pulp.LpProblem("Harmonic", pulp.LpMaximize)

        # Voter assignment variable
        x = pulp.LpVariable.dicts(
            "x", ((i, j, l) for i in range(n) for j in range(m) for l in range(k)), cat="Binary"
        )
        # Candidate 'elected' variable
        y = pulp.LpVariable.dicts("y", range(m), cat="Binary")

        # Objective function:
        problem += pulp.lpSum(
            (1/(l+1)) * B[i, j] * x[i, j, l] for i in range(n) for j in range(m) for l in range(k)
        )

        # Each candidate may only occupy at most one ranked position
        for i in range(n):
            for j in range(m):
                problem += pulp.lpSum(x[i, j, l] for l in range(k)) <= 1
                
        # Each position l occupied exactly one candidate
        for i in range(n):
            for l in range(k):
                problem += pulp.lpSum(x[i, j, l] for j in range(m)) == 1

        # A voter can only be assigned to a candidate if that candidate is elected
        for i in range(n):
            for j in range(m):
                for l in range(k):
                    problem += x[i, j, l] <= y[j]


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
        self._approve_profile(profile, k)
        m, n = profile.shape
        B = borda_matrix(profile, k)

        is_elected = np.zeros(m, dtype=bool)
        voter_assign_scores = np.zeros(n) - 1

        for _ in range(k):
            scores = np.zeros(m)
            for i in range(m):
                if not is_elected[i]:
                    score_gain = np.sum(np.maximum(voter_assign_scores, B[i, :]))
                    scores[i] = score_gain

            # Break ties randomly
            ranking = tiebreak(scores)
            max_cand = ranking[-1]
            is_elected[max_cand] = True
            voter_assign_scores = np.maximum(voter_assign_scores, B[max_cand, :])

        self.objective = np.sum(voter_assign_scores)
        return np.where(is_elected)[0]
    
    
####################################################################################################


class GreedyMonroe(Election):
    """
    Elect k candidates using a greedy approximation to the
    Monroe rule. At every iteration, this rule
    selects a candidate to add to a growing winner set by greedily selecting
    a candidate and a proportionally sized voter set for it, 
    which together give the largest increase to the current assignment scores.
    
    NOTE: For now, this assumes that the number of voters n is divisible by the number of winners k.
    
    For more information, please see:
    "What Do Multiwinner Voting Rules Do? An Experiment Over the Two-Dimensional Euclidean Domain"
    Elkind et al (2019)
    (https://arxiv.org/abs/1901.09217)
    
    
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
        self._approve_profile(profile, k)
        m, n = profile.shape
        B = borda_matrix(profile, k)
        
        if n % k != 0:
            raise ValueError("Number of voters must be divisible by the number of winners.")
        size = n // k

        is_elected = np.zeros(m, dtype=bool)
        voter_assign_mask = np.zeros(n, dtype=bool)

        for _ in range(k):
            scores = np.zeros(m)
            for i in range(m):
                if not is_elected[i]:
                    top_scores = np.sort(B[i, ~voter_assign_mask])[::-1][:size]
                    score_gain = np.sum(top_scores)
                    scores[i] = score_gain

            # Break ties randomly
            ranking = tiebreak(scores)
            max_cand = ranking[-1]
            is_elected[max_cand] = True
            available_voters = np.where(~voter_assign_mask)[0]
            max_cand_voters = tiebreak(B[max_cand, ~voter_assign_mask])[::-1][:size]
            voter_assign_mask[available_voters[max_cand_voters]] = True

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
    voting rule. To really use this as a multiwinner rule, need to construct a special 
    preference profile. Please see the example notebooks for more detail.
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
        
        self._approve_profile(profile, k)
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


class CommitteeVeto(PluralityVeto):
    """
    Elect k candidates with the Committee Veto mechanism. This is really identical to PluralityVeto,
    but should be run with a special preference profile constructed 
    over all possible comittees (Just helpful to have another class with a different name).
    """
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
        self._approve_profile(profile, k)
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
        self._approve_profile(profile, k)        
        m, n = profile.shape
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
        self._approve_profile(profile, k)    
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
        
    def update_indices(self):
        """
        Updates the first place vote indices of voters in the preference profile, to account 
        for candidates who have already been elected.
        """
        for i in range(self.n):
            while self.elected_mask[self.profile[self.voter_indices[i], i]] == 1:
                self.voter_indices[i] += 1
        
    def elect(self, profile : NDArray, k : int) -> NDArray:
        """
        Elect k candidates with the DMRD method.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        self._approve_profile(profile, k)
        self.m, self.n = profile.shape
        self.profile = profile
        self.voter_indices = np.zeros(self.n, dtype=int)
        self.elected_mask = np.zeros(self.m, dtype=bool)
        elected = np.zeros(k, dtype=int) - 1
        voter_probability = np.ones(self.n) / self.n

        for i in range(k):
            dictator = np.random.choice(range(self.n), p=voter_probability)
            winner = self.profile[self.voter_indices[dictator], dictator]
            elected[i] = winner
            self.elected_mask[winner] = True

            # Find who voted for the winner
            first_choice_votes = self.profile[self.voter_indices, np.arange(self.n)]
            mask = first_choice_votes == winner

            # Adjusts voter probability for the next round
            voter_probability[mask] *= self.rho
            voter_probability /= np.sum(voter_probability)

            # Effectively removes winning candidate from profile
            self.update_indices()

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
