import numpy as np
from tools import remove_candidates


def SNTV(profile, k):
    first_choice_votes = profile[0,:]
    cands, counts = np.unique(first_choice_votes, return_counts = True)
    elected = cands[np.argsort(counts)[::-1][:min(k, len(cands))]]
    return elected


def STV(profile, k):
    m,n = profile.shape
    droop = int((n/(k+1)) + 1)
    elected_mask = np.zeros(m)
    eliminated_mask = np.zeros(m)

    # Keeps track of voter's 'position' on the preference profile.
    voter_indices = np.zeros(n, dtype = int)
    # Keeps track of surplus votes to be added over.
    surplus = np.ones(n)

    elected_count = 0
    eliminated_count = 0
    while elected_count < k and eliminated_count < m:
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
                    while (elected_mask[profile[voter_indices[v],v]] == 1) or (eliminated_mask[profile[voter_indices[v],v]] == 1):
                        voter_indices[v] += 1
                        # ballot fully exhausted
                        if voter_indices[v] >= m:
                            print('ballot exhausted')
                            voter_indices[v] = -1
                            break
        
        else:
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
                        while (elected_mask[profile[voter_indices[v],v]] == 1) or (eliminated_mask[profile[voter_indices[v],v]] == 1):
                            voter_indices[v] += 1
                            # ballot fully exhausted
                            if voter_indices[v] >= m:
                                print('ballot exhausted')
                                voter_indices[v] = -1
                                break
                                
                    # only eliminate one candidate per round
                    break

    #if elected_count < k:
    #    elected = np.array([i for i in range(m) if i not in eliminated])

    return np.where(elected_mask == 1)[0]



def Borda(profile, k):
    m,n = profile.shape
    candidate_scores = np.zeros(m)
    
    for i in range(n):
        for j in range(m):
            c = profile[j,i]
            candidate_scores[c] += (m-1) - j
    
    elected = np.argsort(candidate_scores)[::-1][:k]
    return elected
    
    

def RandomDictator(profile,k):
    m,n = profile.shape
    voter_indices = np.zeros(n, dtype = int)
    elected = np.zeros(k, dtype = int) - 1
    
    for i in range(k):
        first_choice_votes = profile[voter_indices, np.arange(n)]
        winner = np.random.choice(first_choice_votes)
        elected[i] = winner
        
        # removes winning candidate from profile
        mask = (first_choice_votes == winner)
        voter_indices[mask] += 1
        
    return elected


def PRD(profile,k, p = None, q = None):
    m,n = profile.shape
    if p is None:
        p = 1 / (m - 1)
    if q is None:
        q = 2
        
    voter_indices = np.zeros(n, dtype = int)
    elected = np.zeros(k, dtype = int) - 1
    for i in range(k):
        first_choice_votes = profile[voter_indices, np.arange(n)]
        coin_flip = np.random.uniform()
        if coin_flip <=p:
            cands,counts = np.unique(first_choice_votes, return_counts = True)
            prob = np.power(counts*1.0, q)
            prob /= np.sum(prob)
            winner = np.random.choice(cands, p = prob)
        else:
            winner = np.random.choice(first_choice_votes)
            
        elected[i] = winner
        # removes winning candidate from profile
        mask = (first_choice_votes == winner)
        voter_indices[mask] += 1
        
    return elected



def PluralityVeto(profile, k):
    m,n = profile.shape
    candidate_scores = np.zeros(m)
    eliminated = np.zeros(m - k) - 1
    eliminated_count = 0
    
    # count initial plurality scores 
    first_choice_votes = profile[0,:]
    for c in first_choice_votes:
        candidate_scores[c] += 1
        
    zero_scores = np.where(candidate_scores == 0)[0]
    if len(zero_scores) > m - k:
        np.random.shuffle(zero_scores)
        zero_scores = zero_scores[:(m-k)]
            
    profile = remove_candidates(profile, zero_scores)
    eliminated[:len(zero_scores)] = zero_scores
    eliminated_count += len(zero_scores)
    
    # veto
    random_order = list(range(n))
    np.random.shuffle(random_order)
    while eliminated_count < (m - k):
        for i,v in enumerate(random_order):
            least_preferred = profile[-1,v]
            candidate_scores[least_preferred] -= 1
            if candidate_scores[least_preferred] <= 0:
                if least_preferred in eliminated:
                    print(profile)
                profile = remove_candidates(profile, [least_preferred])
                eliminated[eliminated_count] = least_preferred
                eliminated_count += 1
                random_order = random_order[i + 1 :] + random_order[: i + 1]
                break
            
    elected = np.array([c for c in range(m) if c not in eliminated])
    return elected
                
    
    
    
    
        
    