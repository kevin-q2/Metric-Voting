import numpy as np
import pulp
from tools import remove_candidates, borda_matrix


def SNTV(profile, k):
    """
    Elect k candidates with the largest plurality scores.
    
    Args:
    profile (np.ndarray): (candidates x voters) Preference Profile. 
    k (int): Number of candidates to elect
    
    Returns:
    elected (np.ndarray): Winning candidates
    """
    first_choice_votes = profile[0,:]
    cands, counts = np.unique(first_choice_votes, return_counts = True)
    elected = cands[np.argsort(counts)[::-1][:min(k, len(cands))]]
    return elected


def Bloc(profile, k):
    """
    Elect k candidates with the largest k-approval scores.
    
    Args:
    profile (np.ndarray): (candidates x voters) Preference Profile. 
    k (int): Number of candidates to elect
    
    Returns:
    elected (np.ndarray): Winning candidates
    """
    first_choice_votes = profile[:k,:]
    cands, counts = np.unique(first_choice_votes, return_counts = True)
    elected = cands[np.argsort(counts)[::-1][:min(k, len(cands))]]
    return elected


def STV(profile, k):
    """
    Elect k candidates with the Single Transferrable Vote election.
    Uses the droop quota with fractional transfer, and breaks ties randomly.
    
    Args:
    profile (np.ndarray): (candidates x voters) Preference Profile. 
    k (int): Number of candidates to elect
    
    Returns:
    elected (np.ndarray): Winning candidates
    """
    m,n = profile.shape
    if m < k:
        raise ValueError('Requested more elected seats than there are candidates!')
    droop = int((n/(k+1)) + 1)
    elected_mask = np.zeros(m)
    eliminated_mask = np.zeros(m)

    # Keeps track of voter's 'position' on the preference profile.
    voter_indices = np.zeros(n, dtype = int)
    # Keeps track of surplus votes to be added over.
    surplus = np.ones(n)

    elected_count = 0
    eliminated_count = 0
    while elected_count < k and eliminated_count < m - k:
        # Count Plurality Scores:
        candidate_scores = np.zeros(m)
        candidate_voters = {c:[] for c in range(m)}
        for i in range(n):
            # if ballot not exhausted, add voter's surplus to their next preferred candidate
            if voter_indices[i] != -1:
                voter_choice = profile[voter_indices[i], i]
                candidate_scores[voter_choice] += surplus[i]
                candidate_voters[voter_choice].append(i)

        # Check for candidates that satisfy droop score and are still in the race:
        satisfies = np.where((candidate_scores >= droop) & (elected_mask != 1) & (eliminated_mask != 1))[0]
        
        # If there are such candidates, elect them.
        if len(satisfies) > 0:
            # Break ties randomly.
            np.random.shuffle(satisfies)
            
            # Elect
            last_elected = []
            for c in satisfies:
                if elected_count < k:
                    elected_mask[c] = 1
                    last_elected.append(c)
                    elected_count += 1

            # Adjust surplus and indices
            for c in last_elected:
                c_voters = candidate_voters[c]
                total_votes = np.sum(surplus[c_voters])
                surplus_votes = total_votes - droop
                surplus[c_voters] = surplus_votes / total_votes

                # move voter's surplus to their next preferred candidate
                for v in c_voters:
                    while voter_indices[v] != -1 and ((elected_mask[profile[voter_indices[v],v]] == 1) or (eliminated_mask[profile[voter_indices[v],v]] == 1)):
                        voter_indices[v] += 1
                        
                        # ballot fully exhausted
                        if voter_indices[v] >= m:
                            voter_indices[v] = -1
                            break
        
        # Otherwise, delete the candidate with the lowest plurality score.
        else:
            # Again breaking ties randomly
            random_tiebreakers = np.random.rand(m)
            structured_array = np.core.records.fromarrays([candidate_scores, random_tiebreakers], 
                                                        names='scores,rand')
            score_sort = np.argsort(structured_array, order=['scores', 'rand'])
            for e in score_sort:
                if (elected_mask[e] == 0) and (eliminated_mask[e] == 0):
                    eliminated_mask[e] = 1
                    eliminated_count += 1
                    e_voters = candidate_voters[e]
                    
                    # move voter's surplus to their next preferred candidate
                    for v in e_voters:
                        while voter_indices[v] != -1 and ((elected_mask[profile[voter_indices[v],v]] == 1) or (eliminated_mask[profile[voter_indices[v],v]] == 1)):
                            voter_indices[v] += 1
                            
                            # ballot fully exhausted
                            if voter_indices[v] >= m:
                                voter_indices[v] = -1
                                break
                                
                    # only eliminate one candidate per round
                    break

    # Final check: Elect any remaining candidates if needed
    remaining_candidates = np.where((elected_mask == 0) & (eliminated_mask == 0))[0]
    if elected_count < k:
        for c in remaining_candidates:
            if elected_count < k:
                elected_mask[c] = 1
                elected_count += 1
    return np.where(elected_mask == 1)[0]



def Borda(profile, k):
    """
    Elect k candidates with the largest Borda scores.
    
    Args:
    profile (np.ndarray): (candidates x voters) Preference Profile. 
    k (int): Number of candidates to elect
    
    Returns:
    elected (np.ndarray): Winning candidates
    """
    m,n = profile.shape
    candidate_scores = np.zeros(m)
    
    for i in range(n):
        for j in range(m):
            c = profile[j,i]
            candidate_scores[c] += (m-1) - j
    
    elected = np.argsort(candidate_scores)[::-1][:k]
    return elected
    
    
def SMRD(profile,k):
    """
    Elect k candidates from k randomly chosen 'dictators'.
    From each dictator elect their first non-elected candidate.
    
    Args:
    profile (np.ndarray): (candidates x voters) Preference Profile. 
    k (int): Number of candidates to elect
    
    Returns:
    elected (np.ndarray): Winning candidates
    """
    m,n = profile.shape
    if n < k:
        raise ValueError('Assumes n >= k')
    voter_indices = np.zeros(n, dtype = int)
    elected = np.zeros(k, dtype = int) - 1
    dictators = np.random.choice(range(n), size = k, replace = False)
    
    for i in range(k):
        dictator = dictators[i]
        winner = profile[voter_indices[dictator], dictator]
        elected[i] = winner
        
        # Find who voted for the winner
        first_choice_votes = profile[voter_indices, np.arange(n)]
        mask = (first_choice_votes == winner)
        
        # Effectively removes winning candidate from profile
        voter_indices[mask] += 1
        
    return elected    



def OMRD(profile,k):
    """
    Chooses a single random dictator and lets them elect their top k 
    preferences.
    
    Args:
    profile (np.ndarray): (candidates x voters) Preference Profile. 
    k (int): Number of candidates to elect
    
    Returns:
    elected (np.ndarray): Winning candidates
    """
    m,n = profile.shape
    dictator = np.random.choice(range(n))
    elected = profile[:k, dictator]
    return elected



def DMRD(profile,k, rho = 1):
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
    m,n = profile.shape
    voter_probability = np.ones(n)/n
    voter_indices = np.zeros(n, dtype = int)
    elected = np.zeros(k, dtype = int) - 1
    
    for i in range(k):
        dictator = np.random.choice(range(n), p = voter_probability)
        winner = profile[voter_indices[dictator], dictator]
        elected[i] = winner
        
        # Find who voted for the winner
        first_choice_votes = profile[voter_indices, np.arange(n)]
        mask = (first_choice_votes == winner)
        
        # Adjusts voter probability for the next round
        voter_probability[mask] *= rho
        voter_probability /= np.sum(voter_probability)
        
        # Effectively removes winning candidate from profile
        voter_indices[mask] += 1
        
    return elected


def PRD(profile,k, p = None, q = None, rho = 1):
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
    
    m,n = profile.shape
    
    # Default setting
    if p is None:
        p = 1 / (m - 1)
    if q is None:
        q = 2
    
    voter_probability = np.ones(n)/n
    voter_indices = np.zeros(n, dtype = int)
    elected = np.zeros(k, dtype = int) - 1
    for i in range(k):
        first_choice_votes = profile[voter_indices, np.arange(n)]
        
        coin_flip = np.random.uniform()
        if coin_flip <= p:
            cands,counts = np.unique(first_choice_votes, return_counts = True)
            prob = np.power(counts*1.0, q)
            prob /= np.sum(prob)
            winner = np.random.choice(cands, p = prob)
        else:
            #winner = np.random.choice(first_choice_votes)
            dictator = np.random.choice(range(n), p = voter_probability)
            winner = profile[voter_indices[dictator], dictator]
            
        elected[i] = winner
        
        # removes winning candidate from profile
        mask = (first_choice_votes == winner)

        # Adjusts voter probability for the next round
        # (If we did random dictator)
        voter_probability[mask] *= 0.8
        voter_probability /= np.sum(voter_probability)
        
        # Effectively removes winning candidate from profile
        voter_indices[mask] += 1
        
    return elected



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
    m,n = profile.shape
    candidate_scores = np.zeros(m)
    eliminated = np.zeros(m - k) - 1
    eliminated_count = 0
    
    # Count initial plurality scores 
    first_choice_votes = profile[0,:]
    for c in first_choice_votes:
        candidate_scores[c] += 1
        
    # Find candidates with 0 plurality score
    zero_scores = np.where(candidate_scores == 0)[0]
    if len(zero_scores) > m - k:
        np.random.shuffle(zero_scores)
        zero_scores = zero_scores[:(m-k)]
            
    # And remove them from the preference profile
    profile = remove_candidates(profile, zero_scores)
    eliminated[:len(zero_scores)] = zero_scores
    eliminated_count += len(zero_scores)
    
    # Veto in a randomize order
    random_order = list(range(n))
    np.random.shuffle(random_order)
    while eliminated_count < (m - k):
        for i,v in enumerate(random_order):
            least_preferred = profile[-1,v]
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
                
    
def ChamberlinCourant(profile,k):
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
    m,n = profile.shape
    B = borda_matrix(profile)
    
    problem = pulp.LpProblem("Chamberlin-Courant", pulp.LpMaximize)

    # Voter assignment variable (voter j gets assigned to candidate i)
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(m) for j in range(n)), cat='Binary')
    # Candidate 'elected' variable
    y = pulp.LpVariable.dicts("y", range(m), cat='Binary')

    # Objective function:
    problem += pulp.lpSum(B[i, j] * x[i, j] for i in range(m) for j in range(n))

    # Each voter is assigned to exactly one candidate
    for j in range(n):
        problem += pulp.lpSum(x[i, j] for i in range(m)) == 1

    # A voter can only be assigned to a candidate if that candidate is elected
    for i in range(m):
        for j in range(n):
            problem += x[i, j] <= y[i]

    # Elect exactly k candidates
    problem += pulp.lpSum(y[i] for i in range(m)) == k

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    elected = np.array([i for i in range(m) if pulp.value(y[i]) == 1])
    return elected


def Monroe(profile,k):
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
    m,n = profile.shape
    B = borda_matrix(profile)
    
    problem = pulp.LpProblem("Chamberlin-Courant", pulp.LpMaximize)

    # Voter assignment variable
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(m) for j in range(n)), cat='Binary')
    # Candidate 'elected' variable
    y = pulp.LpVariable.dicts("y", range(m), cat='Binary')

    # Objective function:
    problem += pulp.lpSum(B[i, j] * x[i, j] for i in range(m) for j in range(n))

    # Each voter is assigned to exactly one candidate
    for j in range(n):
        problem += pulp.lpSum(x[i, j] for i in range(m)) == 1

    # A voter can only be assigned to a candidate if that candidate is elected
    for i in range(m):
        for j in range(n):
            problem += x[i, j] <= y[i]
            
    # Monroe constraint on the size of candidate's voter sets
    for i in range(m):
        problem += pulp.lpSum(x[i, j] for j in range(n)) >= np.floor(n/k) * y[i]
        problem += pulp.lpSum(x[i, j] for j in range(n)) <= np.ceil(n/k) * y[i]

    # Elect exactly k candidates
    problem += pulp.lpSum(y[i] for i in range(m)) == k

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    elected = np.array([i for i in range(m) if pulp.value(y[i]) == 1])
    return elected


def GreedyCC(profile,k):
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
    m,n = profile.shape
    B = borda_matrix(profile)
            
    is_elected = np.zeros(m, dtype = bool)
    voter_assign_scores = np.zeros(n) - 1

    for _ in range(k):
        max_score = -1
        max_cand = -1
        for i in range(m):
            if not is_elected[i]:
                score_gain = np.sum(np.maximum(voter_assign_scores, B[i,:]))
                if score_gain > max_score:
                    max_score = score_gain
                    max_cand = i
                    
        is_elected[max_cand] = True
        voter_assign_scores = np.maximum(voter_assign_scores, B[max_cand,:])
        
    return np.where(is_elected)[0]
        
    