import numpy as np


def election_sample(generator, elections_dict, gen_input, k):
    """
    Given a ballot generator for creating preference profiles
    from randomly generated metric settings and a dictionary containing a set of election methods
    and optional key word arguments, a number of voters, and a number of winning candidates. 
    Randomly creates a profile and then conducts elections with the given mechanisms 
    recording and outputting the results. 

    Args:
        generator (Spatial): Object for creating random preference profiles.
        elections_dict (dict[callable election, dict[str, Any]]): Election mechanism dictionary where keys are
            election mechanisms and their values are dictionaries with any additional key word arguments.
        gen_input (int OR list[int]): Input to generator.generate() (different for Spatial vs GroupSpatial)
        k (int): Number of candidates to elect.

    Returns:
        voter_positions (np.ndarray): Numpy matrix where each row encodes a voters position
            in the metric space. 
        
        candidate_positions (np.ndarray): Numpy matrix where each row encodes a candidate's
            position in the metric space.
            
        winners (dict[str, list[int]]): Dictionary with election names as keys and their corresponding
            winners indices as values. Querying candidate_positions[winners['STV'],:] for example gives
            the winning candidates positions in the metric space. 
            
        voter_labels (list[int]): List with group labels for voters. 
    """
    
    profile, candidate_positions, voter_positions, voter_labels = generator.generate(gen_input)
    winners = {}
    
    for E, params in elections_dict.items():
        elects = E(profile = profile, k = k, **params)
        winners[E.__name__] = elects
        
    return voter_positions, candidate_positions, winners, voter_labels



def samples(s, generator, elections_dict, gen_inputs, k, filename = None, dim = 2):
    """
    For a number of samples, s, sample elections from election_sample()
    and record the results. 

    Args:
        s (int): Number of samples. 
        generator (list[Spatial]): List of spatial objects for creating random preference profiles.
        elections_dict (dict[callable election, dict[str, Any]]): Election mechanism dictionary where keys are
            election mechanisms and their values are dictionaries with any additional key word arguments.
        gen_input (list[int] OR list[list[int]]): List of inputs to generator.generate() 
            (different ways of doing this for Spatial vs GroupSpatial)
        k (int): Number of candidates to elect.
        filename (str, optional): Filename to save results to, optional but if None results
            will not be saved. 
        dim (int, optional): Number of dimensions in a voter or candidate position in the metric space,
            defaults to 2d. 

    Returns:
        voter_positions (np.ndarray): Numpy matrix where each row encodes a voters position
            in the metric space. 
        
        candidate_positions (np.ndarray): Numpy matrix where each row encodes a candidate's
            position in the metric space.
            
        winners (dict[str, list[int]]): Dictionary with election names as keys and their corresponding
            winners indices as values. Querying candidate_positions[winners['STV'],:] for example gives
            the winning candidates positions in the metric space. 
    """
    results_list = []
    for gidx, gen_input in enumerate(gen_inputs):
        m = generator.m
        n = np.sum(gen_input)
        result_dict = {E.__name__:np.zeros((s, k, dim)) for E in elections_dict.keys()}
        result_dict['voters'] = [np.zeros((s, n, dim))]*s
        result_dict['candidates'] = [np.zeros((s, m, dim))]*s
        result_dict['labels'] = [np.zeros(s, n)]*s
        
        for i in range(s):
            V, C, W, vlabels = election_sample(generator, elections_dict, gen_input, k)
            result_dict['voters'][i] = V
            result_dict['candidates'][i] = C
            result_dict['labels'][i] = vlabels
            for name, idxs in W.items():
                if len(idxs) == k:
                    Cx = C[idxs,:]
                elif len(idxs) <= k:
                    diff = k - len(idxs)
                    empties = np.array([[np.nan]*dim]*diff)
                    Cx = np.append(C[idxs,:], empties, axis = 0)
                else:
                    raise ValueError('More than k candidates elected')
                
                    
                result_dict[name][i] = Cx
                
        if not filename is None:
            if len(gen_inputs) > 1:
                np.savez(filename[:-4] + str(gidx) + filename[-4:] , **result_dict)
            else:
                np.savez(filename, **result_dict)

        results_list.append(result_dict)
        
    return results_list
    