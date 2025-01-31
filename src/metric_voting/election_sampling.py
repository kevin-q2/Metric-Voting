import numpy as np
from joblib import Parallel, delayed
from typing import Dict, Callable, Any, List, Tuple
from numpy.typing import NDArray

from .elections import *
from .measurements import euclidean_cost_array, q_cost_array
from .utils import cost_array_to_ranking

import pulp




def election_sample(
    generator : Callable,
    elections_dict : Dict[Callable, Dict[str, Any]],
    generator_input : Dict[str, Any],
    k : int
) -> Tuple[NDArray, NDArray, Dict[str, NDArray], NDArray]:
    """
    Randomly creates a single preference profile, conducts elections, and records the results.
    Given a ballot generator for creating preference profiles
    from randomly generated metric settings and a dictionary containing a set of election methods
    and optional keyword arguments, a number of voters, and a number of winning candidates.
    
    Args:
        generator (Callable, Spatial): Object for generating 
            random preference profiles in metric space.
        elections_dict (dict[Callable Election, dict[str, Any]]): Election mechanism dictionary 
            where keys are election mechanisms and their values are dictionaries with any
            additional keyword arguments.
        generator_input (Dict[str, Any]): keyword arguments for input to generator.generate()
        k (int): Number of candidates to elect.

    Returns:
        winners (dict[str, np.ndarray]): Dictionary with election names as keys and their 
            corresponding winners as values. Winners are shown by a length m boolean array 
            with True values representing winning candidates. 
            Querying candidate_positions[winners['STV'],:] for example gives
            the winning candidates mask.
        candidate_positions (np.ndarray): Numpy matrix where each row encodes a candidate's
            position in the metric space.
        voter_positions (np.ndarray): Numpy matrix where each row encodes a voters position
            in the metric space.
        candidate_labels (NDArray): Length m array of group labels for voters.
        voter_labels (NDArray): Length n array of group labels for voters.
        
    Example Usage and Extra information:
    - A generator object is taken as one of the classes from spatial_generation.py, 
        please see the full documentation there for more specific parameter info.
        
    - Its purpose is to generate random voter/candidate positions in metric space, 
        and will generally take parameters which define the distributions to sample from.
        
    - generator_input, on the other hand, is used specifically for the .generate() method of 
        a generator object. Here it is given as a *single* dictionary containing
        the input voter and candidate group sizes to pass to the .generate() method.
        
    - The elections_dict is a dictionary where the keys are election mechanisms objects 
        from elections.py, and values are dictionaries with any parameters to use 
        when initializing that election class.
    ```
    generator = GroupSpatial(...group spatial params ...)
    elections_dict = {Borda : {}, STV : {'transfer_type' : 'weighted-fractional'}}
    # Input for the generator.generate() method:
    generator_input = {'voter_group_sizes': group_sizes, 'candidate_group_sizes': [m]}
    winners, candidate_positions, voter_positions, candidate_labels, voter_labels = 
        election_sample(
            generator,
            elections_dict,
            generator_input,
            k = 3
        )
    )
    ```
    """
    (profile,
     candidate_positions,
     voter_positions,
     candidate_labels,
     voter_labels) = generator.generate(
        **generator_input
    )
    winners = {}

    for election, params in elections_dict.items():
        if election.__name__ == "CommitteeVeto":
            # Form the multi-winner plurality veto profile.
            # Assuming Euclidean Distance here!!
            if 'q' in params:
                q = params['q']
            else:
                q = k
            
            cost_arr = euclidean_cost_array(voter_positions, candidate_positions)
            candidate_subsets = [set(_) for _ in profile[:k,:].T]
            q_cost_arr = q_cost_array(q, cost_arr, candidate_subsets)
            q_profile = cost_array_to_ranking(q_cost_arr)
            elect_subset = election().elect(profile=q_profile, k=1)[0]
            elects = np.array(list(candidate_subsets[elect_subset]))
            
        else:
            try:
                elects = election(**params).elect(profile=profile, k=k)
            except pulp.apis.core.PulpSolverError:
                elects = np.zeros(k, dtype=int) - 1
                np.save("error_profile.npy", profile)
        
        winners[election.__name__] = elects

    return winners, candidate_positions, voter_positions, candidate_labels, voter_labels





def samples(
    s : int,
    generator : Callable, 
    elections_dict : Dict[Callable, Dict[str, Any]],
    generator_input : List[Dict[str, Any]],
    k : int,
    dim : int = 2,
    filename : str = None
):
    """
    For a given number of samples, sample elections with election_sample()
    and record the results.

    Args:
        s (int): Number of samples.
        generator (Spatial): Spatial object for creating random preference profiles.
        elections_dict (dict[Callable Election, dict[str, Any]]): Election mechanism dictionary 
            where keys are election mechanisms and their values are dictionaries with any additional
            keyword arguments.
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
            
            
    Example Usage and Extra information:
    - A generator object is taken as one of the classes from spatial_generation.py, 
        please see the full documentation there for more specific parameter info.
        
    - Its purpose is to generate random voter/candidate positions in metric space, 
        and will generally take parameters which define the distributions to sample from.
        
    - generator_input, on the other hand, is used specifically for the .generate() method of 
        a generator object. It is a *list* of dictionaries where each dictionary contains
        the input voter and candidate group sizes to pass to the .generate() method.
        For every set of input parameters in the list, a new experiment is performed --
        this is useful for experiments in which the sizes of voter/candidate groups are varied.
        
    - The elections_dict is a dictionary where the keys are election mechanisms objects 
        from elections.py, and values are dictionaries with any parameters to use 
        when initializing that election class.
    ```
    generator = GroupSpatial(...group spatial params ...)
    elections_dict = {Borda : {}, STV : {'transfer_type' : 'weighted-fractional'}}
    # Input for the generator.generate() method:
    generator_input = [{'voter_group_sizes': group_sizes, 'candidate_group_sizes': [m]}]
    sample_result_list = 
        samples(1000, generator, elections_dict, generator_input, k = 3, dim = 2, filename = f)
    )
    ```
    """
    results_list = []
    for gidx, gen_input in enumerate(generator_input):
        try:
            n = gen_input["n"]
            m = gen_input["m"]
        except KeyError:
            n = sum(gen_input["voter_group_sizes"])
            m = sum(gen_input["candidate_group_sizes"])
        

        result_dict = {election.__name__: np.zeros((s, m), dtype = bool)
                       for election in elections_dict.keys()}
        result_dict["voters"] = [np.zeros((s, n, dim))] * s
        result_dict["candidates"] = [np.zeros((s, m, dim))] * s
        result_dict["voter_labels"] = [np.zeros((s, n), dtype = int)] * s
        result_dict["candidate_labels"] = [np.zeros((s, m), dtype = int)] * s

        for i in range(s):
            W, C, V, clabels, vlabels = election_sample(generator, elections_dict, gen_input, k)
            result_dict["voters"][i] = V
            result_dict["candidates"][i] = C
            result_dict["voter_labels"][i] = vlabels
            result_dict["candidate_labels"][i] = clabels
            for name, idxs in W.items():
                winner_mask = np.zeros(m, dtype=bool)
                winner_mask[idxs.astype(np.int32)] = True
                result_dict[name][i] = winner_mask
                

        if not filename is None:
            if len(generator_input) > 1:
                np.savez_compressed(filename[:-4] + str(gidx) + filename[-4:], **result_dict)
            else:
                np.savez_compressed(filename, **result_dict)

        results_list.append(result_dict)

    return results_list


####################################################################################################

def sample_task(generator, elections_dict, gen_input, k):
    W, C, V, clabels, vlabels = election_sample(generator, elections_dict, gen_input, k)
    task_result = {'V': V, 'C': C, 'W': W, 'clabels':clabels, 'vlabels': vlabels}
    return task_result


def parallel_samples(
    s : int,
    generator : Callable, 
    elections_dict : Dict[Callable, Dict[str, Any]],
    generator_input : List[Dict[str, Any]],
    k : int,
    dim : int = 2,
    cpu_count : int = 8,
    filename : str = None
):
    """
    For a given number of samples, sample elections with election_sample()
    and record the results. This version of the sampling function is indentical 
    to the previous aside from the fact that results are computed in parallel.

    Args:
        s (int): Number of samples.
        generator_list (Spatial): List of spatial objects for creating 
            random preference profiles.
        elections_dict (dict[Callable Election, dict[str, Any]]): Election mechanism dictionary 
            where keys are election mechanisms and their values are dictionaries with any additional
            keyword arguments.
        generator_input (List[dict[str, Any]]): Dictionary for input settings to 
            generator.generate().
        k (int): Number of candidates to elect.
        dim (int, optional): Number of dimensions in a voter or candidate position 
            in the metric space, defaults to 2d.
	cpu_count (int, optional): Number of available cpus to use for processing. Defaults to 8.
        filename (str, optional): Filename to save results to, optional but if None results
            will not be saved.

    Returns:
        results_list (List[Dict[str, np.ndarray]]): List of dictionaries where each dictionary
            contains the results of the election sampling
            
            
    Example Usage and Extra information:
    - A generator object is taken as one of the classes from spatial_generation.py, 
        please see the full documentation there for more specific parameter info.
        
    - Its purpose is to generate random voter/candidate positions in metric space, 
        and will generally take parameters which define the distributions to sample from.
        
    - generator_input, on the other hand, is used specifically for the .generate() method of 
        a generator object. It is a *list* of dictionaries where each dictionary contains
        the input voter and candidate group sizes to pass to the .generate() method.
        For every set of input parameters in the list, a new experiment is performed --
        this is useful for experiments in which the sizes of voter/candidate groups are varied.
        
    - The elections_dict is a dictionary where the keys are election mechanisms objects 
        from elections.py, and values are dictionaries with any parameters to use 
        when initializing that election class.
    ```
    generator = GroupSpatial(...group spatial params ...)
    elections_dict = {Borda : {}, STV : {'transfer_type' : 'weighted-fractional'}}
    # Input for the generator.generate() method:
    generator_input = [{'voter_group_sizes': group_sizes, 'candidate_group_sizes': [m]}]
    sample_result_list = 
        samples(1000, generator, elections_dict, generator_input, k = 3, dim = 2, filename = f)
    )
    ```
    """
    results_list = []
    for gidx, gen_input in enumerate(generator_input):
        try:
            n = gen_input["n"]
            m = gen_input["m"]
        except KeyError:
            n = sum(gen_input["voter_group_sizes"])
            m = sum(gen_input["candidate_group_sizes"])
        

        result_dict = {election.__name__: np.zeros((s, m), dtype = bool)
                       for election in elections_dict.keys()}
        result_dict["voters"] = [np.zeros((s, n, dim))] * s
        result_dict["candidates"] = [np.zeros((s, m, dim))] * s
        result_dict["voter_labels"] = [np.zeros((s, n), dtype = int)] * s
        result_dict["candidate_labels"] = [np.zeros((s, m), dtype = int)] * s
        
        results = Parallel(n_jobs=cpu_count, backend = 'threading')(
            delayed(sample_task)(generator, elections_dict, gen_input, k) for _ in range(s)
        )
         
        for i, task_result in enumerate(results):
            W = task_result['W']
            V = task_result['V']
            C = task_result['C']
            clabels = task_result['clabels']
            vlabels = task_result['vlabels']
               
            result_dict["voters"][i] = V
            result_dict["candidates"][i] = C
            result_dict["voter_labels"][i] = vlabels
            result_dict["candidate_labels"][i] = clabels
            for name, idxs in W.items():
                winner_mask = np.zeros(m, dtype=bool)
                winner_mask[idxs.astype(np.int32)] = True
                result_dict[name][i] = winner_mask
                

        if not filename is None:
            if len(generator_input) > 1:
                np.savez_compressed(filename[:-4] + str(gidx) + filename[-4:], **result_dict)
            else:
                np.savez_compressed(filename, **result_dict)

        results_list.append(result_dict)

    return results_list
