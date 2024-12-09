import numpy as np
from multiprocessing import Pool
from typing import Dict, Callable, Any, List, Tuple
from numpy.typing import NDArray

from .measurements import euclidean_cost_array, q_cost_array
from .utils import cost_array_to_ranking





def election_sample(
    generator : Callable,
    elections_dict : Dict[Callable, Dict[str, Any]],
    generator_input : Dict[str, Any],
    k : int
) -> Tuple[NDArray, NDArray, Dict[str, NDArray], NDArray]:
    """
    Randomly creates a profile, conducts elections, and records the results.
    Given a ballot generator for creating preference profiles
    from randomly generated metric settings and a dictionary containing a set of election methods
    and optional key word arguments, a number of voters, and a number of winning candidates.
    
    Args:
        generator (Callable, Spatial): Object for generating 
            random preference profiles in metric space.
        elections_dict (dict[Callable Election, dict[str, Any]]): Election mechanism dictionary 
            where keys are election mechanisms and their values are dictionaries with any
            additional key word arguments.
        generator_input (Dict[str, Any]): Key word arguments for input to generator.generate()
        k (int): Number of candidates to elect.

    Returns:
        voter_positions (np.ndarray): Numpy matrix where each row encodes a voters position
            in the metric space.
        candidate_positions (np.ndarray): Numpy matrix where each row encodes a candidate's
            position in the metric space.
        winners (dict[str, np.ndarray]): Dictionary with election names as keys and their 
            corresponding winners as values. Winners are shown by a length m boolean array 
            with True values representing winning candidates. 
            Querying candidate_positions[winners['STV'],:] for example gives
            the winning candidates mask.
        voter_labels (NDArray): Length n array of group labels for voters.
    """

    profile, candidate_positions, voter_positions, voter_labels = generator.generate(
        **generator_input
    )
    winners = {}

    for E, params in elections_dict.items():
        if E.__name__ == "PluralityVeto":
            # Assuming Euclidean Distance here!!
            cst_array = euclidean_cost_array(voter_positions, candidate_positions)
            candidate_subsets = [set(_) for _ in profile[:k,:].T]
            q_cst_array = q_cost_array(params['q'], cst_array, candidate_subsets)
            q_profile = cost_array_to_ranking(q_cst_array)
            elect_subset = E().elect(profile=q_profile, k=1)[0]
            elects = np.array(list(candidate_subsets[elect_subset]))
            
        else:
            elects = E(**params).elect(profile=profile, k=k)
            
        winners[E.__name__] = elects

    return voter_positions, candidate_positions, winners, voter_labels





def samples(
    s : int,
    generator : List[Callable], 
    elections_dict : Dict[Callable, Dict[str, Any]],
    generator_input : List[Dict[str, Any]],
    k : int,
    dim : int = 2,
    filename : str = None
):
    """
    For a number of samples, s, sample elections from election_sample()
    and record the results.

    Args:
        s (int): Number of samples.
        generator (list[Spatial]): List of spatial objects for creating random preference profiles.
        elections_dict (dict[Callable Election, dict[str, Any]]): Election mechanism dictionary 
            where keys are election mechanisms and their values are dictionaries with any additional
            key word arguments.
        generator_input (List[dict[str, Any]]): Dictionary for input settings to 
            generator.generate().
        k (int): Number of candidates to elect.
        dim (int, optional): Number of dimensions in a voter or candidate position 
            in the metric space, defaults to 2d.
        filename (str, optional): Filename to save results to, optional but if None results
            will not be saved.

    Returns:
        results_list (List[Dict[str, np.ndarray]]): List of dictionaries where each dictionary
            contains the results of the election sampling
    """
    results_list = []
    for gidx, gen_input in enumerate(generator_input):
        try:
            n = gen_input["n"]
            m = gen_input["m"]
        except KeyError:
            n = sum(gen_input["voter_group_sizes"])
            m = sum(gen_input["candidate_group_sizes"])
        

        result_dict = {E.__name__: np.zeros((s, m), dtype = bool) for E in elections_dict.keys()}
        result_dict["voters"] = [np.zeros((s, n, dim))] * s
        result_dict["candidates"] = [np.zeros((s, m, dim))] * s
        result_dict["labels"] = [np.zeros((s, n), dtype = int)] * s

        for i in range(s):
            V, C, W, vlabels = election_sample(generator, elections_dict, gen_input, k)
            result_dict["voters"][i] = V
            result_dict["candidates"][i] = C
            result_dict["labels"][i] = vlabels
            for name, idxs in W.items():
                
                '''
                if len(idxs) == k:
                    Cx = C[idxs, :]
                elif len(idxs) <= k:
                    diff = k - len(idxs)
                    empties = np.array([[np.nan] * dim] * diff)
                    Cx = np.append(C[idxs, :], empties, axis=0)
                else:
                    raise ValueError("More than k candidates elected")
                    
                result_dict[name][i] = Cx
                '''
                mask = np.zeros(m, dtype=bool)
                mask[idxs] = True
                result_dict[name][i] = mask
                

        if not filename is None:
            if len(generator_input) > 1:
                np.savez_compressed(filename[:-4] + str(gidx) + filename[-4:], **result_dict)
            else:
                np.savez_compressed(filename, **result_dict)

        results_list.append(result_dict)

    return results_list


####################################################################################################
# Some (not working!) attempts to parallelize this

'''
class election_sample:
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
    def __init__(self, generator, elections_dict, gen_input, k):
        self.generator = generator
        self.elections_dict = elections_dict
        self.gen_input = gen_input
        self.k = k
    
    def sample(self, i):
        profile, candidate_positions, voter_positions, voter_labels = self.generator.generate(self.gen_input)
        winners = {}
        
        for E, params in self.elections_dict.items():
            elects = E(profile = profile, k = self.k, **params)
            winners[E.__name__] = elects
            
        return voter_positions, candidate_positions, winners, voter_labels


def sample_task(generator, elections_dict, gen_input, k):
    V, C, W, vlabels = election_sample(generator, elections_dict, gen_input, k)
    task_result = {'V': V, 'C': C, 'W': W, 'vlabels': vlabels}
    return task_result
'''


"""
sampler = election_sample(generator, elections_dict, gen_input, k)
with Pool(cpu_count) as p:
    #task_results = p.starmap(sample_task, pool_args)
    task_results = p.map(sampler.sample, range(s))
"""
