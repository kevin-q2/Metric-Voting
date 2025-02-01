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
            Function of the form f(m, k, i) where m is the total number of candidates,
            k is the voter's ranking of the candidate, and
            i is a voter's ranking in [1,...,m] of the candidate.
    """
    def __init__(self, scoring_scheme: Callable[[int, int, int], float]):        
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
    which uses the scoring scheme f(m, k, i) = 1 for i <= k and 0 otherwise. In other words, 
    all candidates in the top k positions of a voter's ranking receive a score of 1, and all
    other candidates receive a score of 0.
    """
    def __init__(self):
        scoring_scheme = lambda x, y, z: 1 if z <= y else 0
        super().__init__(scoring_scheme)
    

####################################################################################################


class Borda(CommitteeScoring):
    """
    Elect k candidates with the Borda Scoring Mechanism. This is a committee scoring method 
    which uses the scoring scheme f(m, k, i) = m - (i + 1). In other words, candidates receive 
    scores of m - 1 for being ranked first, m - 2 for being ranked second, and so on.
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
        voter_current_ballot_position (NDArray): Tracks voter's next available choices.
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
        self.voter_current_ballot_position = None
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
        self.voter_current_ballot_position = np.zeros(self.n, dtype=int)
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
                print("voter indices: " + str(self.voter_current_ballot_position))
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
            if self.voter_current_ballot_position[i] != -1:
                voter_choice = self.profile[self.voter_current_ballot_position[i], i]
                
                if self.voter_weights[i] > 0:
                    #candidate_scores[voter_choice] += self.voter_weights[i]
                    candidate_voters[voter_choice].append(i)

        for c, voters in candidate_voters.items():
            candidate_scores[c] = np.sum(self.voter_weights[voters])
            
        # Finds cands that satisfy the droop quota that are not elected or eliminated.
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
            total_votes (float): Total number of votes for the candidate.
            candidate_voters (List[int]): List of indices for voters
                who voted for the elected candidate.
        """
        surplus_votes = total_votes - self.droop
        
        if self.verbose:
            print("surplus: " + str(surplus_votes))
        
        # This is done with tolerance to account for floating point issues,
        # which are a result of fractional surplus votes. 
        if surplus_votes > 1e-10:
            
            # We adjust the weight of the voters whose top remaining choice
            # was the elected candidate so that their next candidate gets 
            # votes equal to some fraction of their initial weight,
            # where the fraction depends on a function of the surplus votes.
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
            
        # Once the weights have been adjusted, we move the voters'
        # ballot positions to complete the transfer.
        self.adjust_current_ballot_positions(candidate_voters)
        
        
    def adjust_current_ballot_positions(self, candidate_voters : List[int]):
        """
        Adjust voter indices to the next available candidate on their ballot.
        If a voter's ballot has been exhausted, remove them from any further
        voting process by setting their voter index to -1.
        
        Args:
            candidate_voters (List[int]): List of voter indices for 
                voters who voted for the candidate to transfer from.
        """
        for v in candidate_voters:
            while self.voter_current_ballot_position[v] != -1 and (
                (self.elected_mask[self.profile[self.voter_current_ballot_position[v], v]] == 1)
                or (self.eliminated_mask[self.profile[self.voter_current_ballot_position[v], v]] == 1)
            ):
                self.voter_current_ballot_position[v] += 1

                # ballot fully exhausted
                if self.voter_current_ballot_position[v] >= self.m:
                    self.voter_current_ballot_position[v] = -1
                    break
 

####################################################################################################


class ChamberlinCourant(Election):
    """
    Elect k candidates with the Chamberlain Courant Mechanism.
    This function uses an integer linear program to compute an optimal
    assignment of voters to candidates to maximize the assignment scores
    (where assignment scores are calculated with the borda score).
    
    This is based on work from the following papers:
    "Achieving Fully Proportional Representation: Approximability Results"
    Skowron et al (2013)
    (https://arxiv.org/abs/1312.4026)
    NOTE: See section 4.7 starting on page 32.
    
    "What Do Multiwinner Voting Rules Do? An Experiment Over the Two-Dimensional Euclidean Domain"
    Elkind et al (2019)
    (https://arxiv.org/abs/1901.09217)
    
    LP definition:
    - Let B be the Borda matrix of the profile, where B(i, j) is the Borda score of candidate j 
        in voter i's ranking.
    - Let x(i, j) be a binary representation variable indicating if voter i is assigned 
        candidate j as its representative.
    - Let y(j) be a binary variable indicating if candidate j is elected.    
    
    Objective:    
        max: sum_{over all voters i and candidates j} Borda-Sore(i, j) * x(i, j)
    
    subject to:
        1) sum_{over all candidates j} x(i, j) = 1 for all i 
            (each voter must get exactly one representative)
            
        2)  x(i, j) <= y(j) for all i, j
            (a voter can only be assigned to a candidate if that candidate is elected)
            
        3) sum_{over all candidates j} y(j) = k
            (exactly k candidates are elected)

    
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
        
        # Randomly permute the candidates to break ties
        candidate_permutation = np.random.permutation(m)
        B = B[:, candidate_permutation]

        problem = pulp.LpProblem("Chamberlin-Courant", pulp.LpMaximize)

        # Voter assignment variable (voter i gets assigned to candidate j)
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
        self.objective = pulp.value(problem.objective)
        elected = np.array([j for j in range(m) if pulp.value(y[j]) == 1])
        # Translate back to original indices:
        elected = candidate_permutation[elected]
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
    
    This is based on work from the following papers:
    "Achieving Fully Proportional Representation: Approximability Results"
    Skowron et al (2013)
    (https://arxiv.org/abs/1312.4026)
    NOTE: See section 4.7 starting on page 32.
    
    "What Do Multiwinner Voting Rules Do? An Experiment Over the Two-Dimensional Euclidean Domain"
    Elkind et al (2019)
    (https://arxiv.org/abs/1901.09217)
    
    
    LP definition:
    - Let B be the Borda matrix of the profile, where B(i, j) is the Borda score of candidate j 
        in voter i's ranking.
    - Let x(i, j) be a binary representation variable indicating if voter i is assigned 
        candidate j as its representative.
    - Let y(j) be a binary variable indicating if candidate j is elected.    
    
    Objective:    
        max: sum_{over all voters i and candidates j} Borda-Sore(i, j) * x(i, j)
    
    subject to:
        1) sum_{over all candidates j} x(i, j) = 1 for all i 
            (each voter must get exactly one representative)
            
        2)  x(i, j) <= y(j) for all i, j
            (a voter can only be assigned to a candidate if that candidate is elected)
            
        3) sum_{over all candidates j} y(j) = k
            (exactly k candidates are elected)
            
        4) floor(n/k) * y(j) <= sum_{over all voters i} x(i, j) <= ceiling(n/k) * y(j) for all j
            (*Monroe constraint* -- each candidate can only represent a certain number of voters)
    
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
        
        # Randomly permute the candidates to break ties
        candidate_permutation = np.random.permutation(m)
        B = B[:, candidate_permutation]

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
        self.objective = pulp.value(problem.objective)
        elected = np.array([j for j in range(m) if pulp.value(y[j]) == 1])
        # Translate back to original indices:
        elected = candidate_permutation[elected]
        return elected
    
    
####################################################################################################


class PAV(Election):
    """
    NOTE: This is not working as expected right now ... I will plan to fix it as soon as possible.
    
    Elect k candidates with the harmonic comittee scoring mechanism..
    This function uses an integer linear program to find a winner set which
    maximizes the sum of harmonic adjusted committee scores.
    With an approval scoring scheme, this is equivalent to the PAV rule.
    With a borda scoring scheme, this is equivalent to the Harmonic Borda rule.
    
    For more information, please see:
    "What Do Multiwinner Voting Rules Do? An Experiment Over the Two-Dimensional Euclidean Domain"
    Elkind et al (2019)
    (https://arxiv.org/abs/1901.09217)
    NOTE: See section E in the appendix (it is labeled as harmonic borda there, but 
        is the same linear program, just with a different scoring mechanism).
    
    NOTE: Define linear program here once this is fixed.
    
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
        
        # Randomly permute the candidates to break ties
        #candidate_permutation = np.random.permutation(m)
        #B = B[:, candidate_permutation]

        problem = pulp.LpProblem("PAV", pulp.LpMaximize)

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
                # Voter  i's jth ranked candidate:
                #c = np.where(candidate_permutation == profile[j, i])[0][0]
                c = profile[j, i]
                for l in range(k):
                    problem += x[i, j, l] <= y[c]

        # Elect exactly k candidates
        problem += pulp.lpSum(y[j] for j in range(m)) == k

        problem.solve(self.solver)
        self.objective = pulp.value(problem.objective)
        elected = np.array([j for j in range(m) if pulp.value(y[j]) == 1])
        # Translate back to original indices:
        #elected = candidate_permutation[elected]
        return elected
    
    
####################################################################################################


class GreedyCC(Election):
    """
    Elect k candidates using a greedy approximation to the
    Chamberlain Courant rule. At every iteration, this rule
    selects a candidate to add to a growing winner set by greedily selecting
    the candidate which gives the best increase to the current assignment scores.
    
    For more information, please see the papers:
    "Budgeted Social Choice: From Consensus to Personalized Decision Making" 
    - Lu and Boutilier (2011)
    (https://www.cs.toronto.edu/~tl/papers/LuBoutilier_budgeted_IJCAI11.pdf)
    
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

            # Break ties randomly amongst candidates who are not yet elected:
            not_elected = np.where(~is_elected)[0]
            ranking = tiebreak(scores[not_elected])
            max_cand = not_elected[ranking[-1]]
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
                
    
    def initialize(self, k):
        """
        Counts first place votes and eliminates any candidates with 0.
        
        Args:
            k (int): Number of candidates to elect.
        """
        # Count initial plurality scores
        first_choice_votes = self.profile[:k, :]
        mentions, counts = np.unique(first_choice_votes, return_counts = True)
        for i,c in enumerate(mentions):
            self.candidate_scores[c] += counts[i]

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
        
        self.initialize(k)
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
        
    
    def approval_round(self, t : int):
        """
        Conduct an approval round of the expanding approvals rule.
        
        Args:
            t (int): Round number.
        """
        for v in self.random_order:
            if self.uncovered_mask[v]:
                c = self.profile[t, v]
                if not self.elected_mask[c]:
                    self.neighborhood[c, v] = 1
                    self.candidate_check_elect(c)
        
    
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

        # Main election loop
        for t in range(m):
            self.random_order = np.random.permutation(n)
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
        dictators = np.random.choice(range(n), size=k, replace=True)

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
            while self.elected_mask[self.profile[self.voter_current_ballot_position[i], i]] == 1:
                self.voter_current_ballot_position[i] += 1
        
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
        self.voter_current_ballot_position = np.zeros(self.n, dtype=int)
        self.elected_mask = np.zeros(self.m, dtype=bool)
        elected = np.zeros(k, dtype=int) - 1
        voter_probability = np.ones(self.n) / self.n

        for i in range(k):
            dictator = np.random.choice(range(self.n), p=voter_probability)
            winner = self.profile[self.voter_current_ballot_position[dictator], dictator]
            elected[i] = winner
            self.elected_mask[winner] = True

            # Find who voted for the winner
            first_choice_votes = self.profile[self.voter_current_ballot_position, np.arange(self.n)]
            support = first_choice_votes == winner
            others = ~support
            
            if np.sum(others) > 0:
                renorm_factor = np.sum(voter_probability[support])/np.sum(voter_probability[others])
                renorm_factor = 1 + (1 - self.rho) * renorm_factor 

                # Adjusts voter probability for the next round
                voter_probability[support] *= self.rho
                voter_probability[others] *= renorm_factor
                #voter_probability /= np.sum(voter_probability)

            # Effectively removes winning candidate from profile
            self.update_indices()

        return elected


####################################################################################################
