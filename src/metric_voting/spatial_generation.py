import numpy as np
import warnings
from .measurements import cost_array
from .utils import euclidean_distance, cost_array_to_ranking, tiebreak
from numpy.typing import NDArray
from typing import Callable, Dict, Tuple, Any, List, Optional, Union


class Spatial:
    """
    Spatial model for ballot generation with distinct voter blocs.
    In some metric space determined by an input distance function, 
    randomly sample each voter's and each candidate's positions 
    from input voter and candidate distributions.
    Using generate() outputs a ranked profile which is consistent
    with the sampled positions (respects distances). 

    Args:
        voter_dist_fn (Callable[..., np.ndarray], optional): Distribution to sample a single
            voter's position from, defaults to uniform distribution.
        voter_dist_fn_params: (dict[str, Any], optional): Parameters to be passed to
            voter_dist_fn, defaults to None,
            which creates the unif(0,1) distribution in 2 dimensions.
        candidate_dist_fn: (Callable[..., np.ndarray], optional): Distribution to sample a
            single candidate's position from, defaults to uniform distribution.
        candidate_dist_fn_params: (dict[str, Any], optional): Parameters to be passed
            to candidate_dist_fn, defaults to None, which creates the unif(0,1)
            distribution in 2 dimensions.
        distance_fn: (Callable[[np.ndarray, np.ndarray], float]], optional):
            Computes distance between a voter and a candidate,
            defaults to euclidean distance.

    Attributes:
        voter_dist_fn (Callable[..., np.ndarray], optional): Distribution to sample a single
            voter's position from, defaults to uniform distribution.
        voter_dist_fn_params: (dict[str, Any], optional): Parameters to be passed to
            voter_dist_fn, defaults to None, 
            which creates the unif(0,1) distribution in 2 dimensions.
        candidate_dist_fn: (Callable[..., np.ndarray], optional): Distribution to sample a
            single candidate's position from, defaults to uniform distribution.
        candidate_dist_fn_params: (dict[str, Any], optional): Parameters to be passed
            to candidate_dist_fn, defaults to None, which creates the unif(0,1)
            distribution in 2 dimensions.
        distance_fn: (Callable[[np.ndarray, np.ndarray], float]], optional):
            Computes distance between a voter and a candidate,
            defaults to euclidean distance.
    """

    def __init__(
        self,
        voter_dist_fn: Callable = np.random.uniform,
        voter_dist_fn_params: Dict[str, Any] = None,
        candidate_dist_fn: Callable = np.random.uniform,
        candidate_dist_fn_params: Dict[str, Any] = None,
        distance_fn: Callable =euclidean_distance,
    ):

        self.voter_dist_fn = voter_dist_fn
        self.candidate_dist_fn = candidate_dist_fn

        if voter_dist_fn_params is None:
            if voter_dist_fn is np.random.uniform:
                self.voter_dist_fn_params = {"low": 0.0, "high": 1.0, "size": 2.0}
            else:
                raise ValueError(
                    "No parameters were given for the input voter distribution."
                )
        else:
            try:
                self.voter_dist_fn(**voter_dist_fn_params)
            except TypeError:
                raise TypeError("Invalid parameters for the voter distribution.")

            self.voter_dist_fn_params = voter_dist_fn_params

        if candidate_dist_fn_params is None:
            if candidate_dist_fn is np.random.uniform:
                self.candidate_dist_fn_params = {"low": 0.0, "high": 1.0, "size": 2.0}
            else:
                raise ValueError(
                    "No parameters were given for the input candidate distribution."
                )
        else:
            try:
                self.candidate_dist_fn(**candidate_dist_fn_params)
            except TypeError:
                raise TypeError("Invalid parameters for the candidate distribution.")

            self.candidate_dist_fn_params = candidate_dist_fn_params

        try:
            v = self.voter_dist_fn(**self.voter_dist_fn_params)
            c = self.candidate_dist_fn(**self.candidate_dist_fn_params)
            distance_fn(v, c)
        except TypeError:
            raise ValueError(
                "Distance function is invalid or incompatible "
                "with voter/candidate distributions."
            )

        self.distance_fn = distance_fn


    def generate(self, n: int, m: int) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Samples a metric position for n voters from
        the voter distribution. Samples a metric position for each candidate
        from the input candidate distribution. With sampled
        positions, this method then creates a ranked preference profile in which
        voter's preferences are consistent with their distances to the candidates
        in the metric space.

        Args:
            n (int): _Number of voters.
            m (int): Number of candidates. 
            
        Returns:
            profile (NDArray): A m x n preference profile with each column j containing 
                voter j's rankings for each of the m candidates.
            candidate_positions (NDArray): A m x d matrix where each row i contains the
                position of candidate i in metric space. 
            voter_positions (NDArray): A n x d matrix where each row i contains the
                position of voter i in metric space.
            candidate_labels (NDArray): A length m array where each element i contains the
                group label for voter i.
            voter_labels (NDArray): A length n array where each element i contains the
                group label for voter i.
        """
        # Sample Candidate Positions
        candidate_positions = np.array(
            [self.candidate_dist_fn(**self.candidate_dist_fn_params) for _ in range(m)]
        )
        
        # Sample Voter Positions
        voter_positions = np.array(
            [self.voter_dist_fn(**self.voter_dist_fn_params) for _ in range(n)]
        )

        candidate_labels = np.zeros(m)
        voter_labels = np.zeros(n)
        
        # Compute Preference Profile
        cst_array = cost_array(voter_positions, candidate_positions, self.distance_fn)
        profile = cost_array_to_ranking(cst_array)

        return profile, candidate_positions, voter_positions, candidate_labels, voter_labels


class GroupSpatial:
    """
    Spatial model for ballot generation with distinct voter blocs.
    In some metric space determined by an input distance function, 
    randomly sample each voter's and each candidate's positions 
    from input voter and candidate distributions.
    Using generate() outputs a ranked profile which is consistent
    with the sampled positions (respects distances). 
    
    This model differs from the previous Spatial model by allowing 
    for different distribution functions for separate blocs of voters,
    as well as different distributions functions for groups of candidates.

    Args:
        n_voter_groups (int): Number of voter groups.
        n_candidate_groups (int): Number of candidate groups. 
        voter_dist_fns (list[Callable[..., np.ndarray]], optional): Distributions to sample
            voter's positions from, defaults to uniform distribution.
        voter_dist_fn_params: (list[dict[str, Any]], optional): Parameters to be passed to
            voter_dist_fns, defaults to None, 
            which creates the unif(0,1) distribution in 2 dimensions.
        candidate_dist_fns (list[Callable[..., np.ndarray]], optional): Distributions to sample
            candidate's positions from, defaults to uniform distribution.
        candidate_dist_fn_params (list[dict[str, Any]], optional): Parameters to be passed
            to candidate_dist_fn, defaults to None, which creates the unif(0,1)
            distribution in 2 dimensions.
        distance_fn: (Callable[[np.ndarray, np.ndarray], float]], optional):
            Computes distance between a voter and a candidate,
            defaults to euclidean distance.

    Attributes:
        n_voter_groups (int): Number of voter groups.
        n_candidate_groups (int): Number of candidate groups. 
        voter_dist_fns (list[Callable[..., np.ndarray]], optional): Distribution to sample a single
            voter's position from, defaults to uniform distribution.
        voter_dist_fn_params: (list[dict[str, Any]], optional): Parameters to be passed to
            voter_dist_fns, defaults to None, 
            which creates the unif(0,1) distribution in 2 dimensions.
        candidate_dist_fns: (Callable[..., np.ndarray], optional): Distribution to sample a
            single candidate's position from, defaults to uniform distribution.
        candidate_dist_fn_params: (Optional[Dict[str, Any]], optional): Parameters to be passed
            to candidate_dist_fn, defaults to None, which creates the unif(0,1)
            distribution in 2 dimensions.
        distance_fn: (Callable[[np.ndarray, np.ndarray], float]], optional):
            Computes distance between a voter and a candidate,
            defaults to euclidean distance.
    """

    def __init__(
        self,
        n_voter_groups: (int),
        n_candidate_groups: (int),
        voter_dist_fns: List[Callable] = None,
        voter_dist_fn_params: List[Dict[str, Any]] = None,
        candidate_dist_fns: List[Callable] = None,
        candidate_dist_fn_params: List[Dict[str, Any]] =None,
        distance_fn: Callable =euclidean_distance,
    ):

        self.n_voter_groups = n_voter_groups
        self.n_candidate_groups = n_candidate_groups

        if voter_dist_fns is None:
            self.voter_dist_fns = [np.random.uniform] * n_voter_groups
        else:
            # Peter Note: Check to make sure that the voter distribution callables have the
            # correct signature
            
            # Kevin: One issue is that I don't expect 
            # the voter_dist_fns to have always have a specific signature. For example, 
            # np.random.uniform and np.random.normal are a bit different.
            # I could check to make sure that the signature matches what's in 
            # voter_dist_fn_params, but is there an advantage to doing all this 
            # rather than just attempting to call the function with the parameters 
            # like I do below?
            
            if len(voter_dist_fns) != n_voter_groups:
                raise ValueError(
                    "Group size does not match with given voter distributions."
                )

            self.voter_dist_fns = voter_dist_fns

        if candidate_dist_fns is None:
            self.candidate_dist_fns = [np.random.uniform] * n_candidate_groups
        else:
            # Peter Note: Check to make sure that the candidate distribution callables have the
            # correct signature
            
            # Kevin: Same as above. 
            
            if len(candidate_dist_fns) != n_candidate_groups:
                raise ValueError(
                    "Group size does not match with given candidate distributions."
                )

            self.candidate_dist_fns = candidate_dist_fns

        if voter_dist_fn_params is None:
            self.voter_dist_fn_params = [{} for _ in range(n_voter_groups)]
            for i, dist in enumerate(self.voter_dist_fns):
                if dist is np.random.uniform:
                    self.voter_dist_fn_params[i] = {"low": 0.0, "high": 1.0, "size": 2}
                else:
                    raise ValueError(
                        "No parameters were given for the input voter distribution."
                    )
                # Peter Note: Check to make sure that the kwargs are valid params for the
                # callables used in the distribution function
                
                # Kevin: If I understand correctly, this is what the next block of code does.
        else:
            for i, dist in enumerate(self.voter_dist_fns):
                try:
                    dist(**voter_dist_fn_params[i])
                except TypeError as e:
                    raise TypeError((f"Invalid parameters for the voter distribution"
                                    f"in position {i} of the voter_dist_fns list. "
                                    f"Error given: [{e}]"))

            self.voter_dist_fn_params = voter_dist_fn_params

        if candidate_dist_fn_params is None:
            self.candidate_dist_fn_params = [{} for _ in range(n_candidate_groups)]
            for i, dist in enumerate(self.candidate_dist_fns):
                if dist is np.random.uniform:
                    self.candidate_dist_fn_params[i] = {"low": 0.0, "high": 1.0, "size": 2}
                else:
                    raise ValueError(
                        "No parameters were given for the input candidate distribution."
                    )
                # Peter Note: Check to make sure that the kwargs are valid params for the
                # callables used in the distribution function
                
                # Kevin: Same as above.
        else:
            for i, dist in enumerate(self.candidate_dist_fns):
                try:
                    dist(**candidate_dist_fn_params[i])
                except TypeError as e:
                    raise TypeError((f"Invalid parameters for the candidate distribution"
                                    f"in position {i} of the candidate_dist_fns list."
                                    f"Error given: [{e}]"))

            self.candidate_dist_fn_params = candidate_dist_fn_params

        try:
            for i in range(self.n_voter_groups):
                for j in range(self.n_candidate_groups):
                    v = self.voter_dist_fns[i](**self.voter_dist_fn_params[i])
                    c = self.candidate_dist_fns[j](**self.candidate_dist_fn_params[j])
                    distance_fn(v, c)

        except TypeError:
            raise ValueError(
                "Distance function is invalid or incompatible "
                "with voter/candidate distributions."
            )

        self.distance_fn = distance_fn


    def generate(
        self,
        voter_group_sizes : Union[List[int], NDArray], 
        candidate_group_sizes : Union[List[int], NDArray],
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        
        """
        Samples metric positions groups of voters and groups of candidates from
        the input voter and candidate distribution functions.
        
        Once sampled, this method then creates a ranked preference profile in which
        voter's preferences are consistent with their distances to the candidates
        in the metric space.

        Args:
            voter_group_sizes (List[int] OR NDArray): List describing the sizes
                for each voter group.
            candidate_group_sizes (list[int] OR NDArray): List describing the sizes
                for each candidate group.
                
        Returns:
            profile (NDArray): A m x n preference profile with each column j containing 
                voter j's rankings for each of the m candidates.
            candidate_positions (NDArray): A m x d matrix where each row i contains the
                position of candidate i in metric space. 
            voter_positions (NDArray): A n x d matrix where each row i contains the
                position of voter i in metric space.
            candidate_labels (NDArray): A length m array where each element i contains the
                group label for voter i.
            voter_labels (NDArray): A length n array where each element i contains the
                group label for voter i.
        """

        if len(voter_group_sizes) != self.n_voter_groups:
            raise ValueError("Number of voter groups is inconsistent with n_voter_groups.")
        if len(candidate_group_sizes) != self.n_candidate_groups:
            raise ValueError("Number of candidate groups is inconsistent with n_candidate_groups.")
        
        n = sum(voter_group_sizes)
        m = sum(candidate_group_sizes)

        # Sample Candidate Positions
        candidate_positions = [
            [
                self.candidate_dist_fns[i](**self.candidate_dist_fn_params[i])
                for _ in range(candidate_group_sizes[i])
            ]
            for i in range(self.n_candidate_groups)
            if candidate_group_sizes[i] != 0
        ]
        candidate_positions = np.vstack(candidate_positions)
        
        candidate_labels = [[i] * candidate_group_sizes[i] for i in range(self.n_candidate_groups)]
        candidate_labels = np.array([item for sublist in candidate_labels for item in sublist])
        
        # Sample Voter Positions
        voter_positions = [
            [
                self.voter_dist_fns[i](**self.voter_dist_fn_params[i])
                for _ in range(voter_group_sizes[i])
            ]
            for i in range(self.n_voter_groups)
            if voter_group_sizes[i] != 0
        ]
        voter_positions = np.vstack(voter_positions)

        voter_labels = [[i] * voter_group_sizes[i] for i in range(self.n_voter_groups)]
        voter_labels = np.array([item for sublist in voter_labels for item in sublist])

        # Compute Preference Profile
        # (change to euclidean cost array if you want to do this really fast)
        cst_array = cost_array(voter_positions, candidate_positions, self.distance_fn)
        profile = cost_array_to_ranking(cst_array)


        return profile, candidate_positions, voter_positions, candidate_labels, voter_labels




class ProbabilisticGroupSpatial(GroupSpatial):
    """
    Spatial model for ballot generation with distinct voter blocs.
    In some metric space determined by an input distance function, 
    randomly sample each voter's and each candidate's positions 
    from input voter and candidate distributions.
    Using generate() outputs a ranked profile which is consistent
    with the sampled positions (respects distances). 
    
    This model differs from the previous GroupSpatial model 
    by randomly assigning voters and candidates to groups rather 
    than starting with deterministically sized groups.

    Args:
        n_voter_groups (int): Number of voter groups.
        n_candidate_groups (int): Number of candidate groups. 
        voter_dist_fns (list[Callable[..., np.ndarray]], optional): Distributions to sample
            voter's positions from, defaults to uniform distribution.
        voter_dist_fn_params: (list[dict[str, Any]], optional): Parameters to be passed to
            voter_dist_fns, defaults to None,
            which creates the unif(0,1) distribution in 2 dimensions.
        candidate_dist_fns (list[Callable[..., np.ndarray]], optional): Distributions to sample
            candidate's positions from, defaults to uniform distribution.
        candidate_dist_fn_params (list[dict[str, Any]], optional): Parameters to be passed
            to candidate_dist_fn, defaults to None, which creates the unif(0,1)
            distribution in 2 dimensions.
        distance_fn: (Callable[[np.ndarray, np.ndarray], float]], optional):
            Computes distance between a voter and a candidate,
            defaults to euclidean distance.

    Attributes:
        n_voter_groups (int): Number of voter groups.
        n_candidate_groups (int): Number of candidate groups. 
        voter_dist_fns (list[Callable[..., np.ndarray]], optional): Distribution to sample a single
            voter's position from, defaults to uniform distribution.
        voter_dist_fn_params: (list[dict[str, Any]], optional): Parameters to be passed to
            voter_dist_fns, defaults to None, 
            which creates the unif(0,1) distribution in 2 dimensions.
        candidate_dist_fns: (Callable[..., np.ndarray], optional): Distribution to sample a
            single candidate's position from, defaults to uniform distribution.
        candidate_dist_fn_params: (Optional[Dict[str, Any]], optional): Parameters to be passed
            to candidate_dist_fn, defaults to None, which creates the unif(0,1)
            distribution in 2 dimensions.
        distance_fn: (Callable[[np.ndarray, np.ndarray], float]], optional):
            Computes distance between a voter and a candidate,
            defaults to euclidean distance.
    """

    def __init__(
        self,
        n_voter_groups: (int),
        n_candidate_groups: (int),
        voter_dist_fns: List[Callable] = None,
        voter_dist_fn_params: List[Dict[str, Any]] = None,
        candidate_dist_fns: List[Callable] = None,
        candidate_dist_fn_params: List[Dict[str, Any]] =None,
        distance_fn: Callable =euclidean_distance,
    ):
    
        super().__init__(
            n_voter_groups, 
            n_candidate_groups, 
            voter_dist_fns,
            voter_dist_fn_params, 
            candidate_dist_fns, 
            candidate_dist_fn_params, 
            distance_fn
        )
        
    
    def generate(
        self,
        n : int,
        m : int,
        voter_group_probs : Union[List[float], NDArray], 
        candidate_group_probs : Union[List[float], NDArray]
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        
        """
        Samples metric positions groups of voters and groups of candidates from
        the input voter and candidate distribution functions.
        
        Once sampled, this method then creates a ranked preference profile in which
        voter's preferences are consistent with their distances to the candidates
        in the metric space.

        Args:
            n (int): Number of voters.
            m (int): Number of candidates.
            voter_group_probs (Union[List[float], NDArray]): List describing the 
                probability of belonging to each voter group.
            candidate_group_sizes (Union[List[float], NDArray]): List describing the
                probability of belonging to each candidate group.
                
        Returns:
            profile (NDArray): A m x n preference profile with each column j containing 
                voter j's rankings for each of the m candidates.
            candidate_positions (NDArray): A m x d matrix where each row i contains the
                position of candidate i in metric space. 
            voter_positions (NDArray): A n x d matrix where each row i contains the
                position of voter i in metric space.
            candidate_labels (NDArray): A length m array where each element i contains the
                group label for voter i.
            voter_labels (NDArray): A length n array where each element i contains the
                group label for voter i.
        """
        
        if len(voter_group_probs) != self.n_voter_groups:
            raise ValueError("Number of voter probabilities is inconsistent with n_voter_groups.")
        if len(candidate_group_probs) != self.n_candidate_groups:
            raise ValueError("Number of candidate probabilities "
                             "is inconsistent with n_candidate_groups.")
        
        if sum(voter_group_probs) != 1:
            raise ValueError("Voter group probabilities do not sum to 1.")
        if sum(candidate_group_probs) != 1:
            raise ValueError("Candidate group probabilities do not sum to 1.")
        
        voter_labels = np.random.choice(self.n_voter_groups, n, p=voter_group_probs)
        # Captures the possibility of having empty groups
        voter_group_sizes = [np.sum(voter_labels == i) for i in range(self.n_voter_groups)]
        
        candidate_labels = np.random.choice(self.n_candidate_groups, m,
                                                  p=candidate_group_probs)
        # Captures the possibility of having empty groups
        candidate_group_sizes = [np.sum(candidate_labels == i)
                                 for i in range(self.n_candidate_groups)]
        
        (profile, candidate_positions, voter_positions,
         candidate_labels, voter_labels) = super().generate(
             voter_group_sizes,
            candidate_group_sizes
        )
        return profile, candidate_positions, voter_positions, candidate_labels, voter_labels
    
    

class RankedSpatial:
    """
    NOTE: This class/method of spatial generation is still a work in progress. It has not 
    been thorougly tested yet! I hope to get to this soon!
    
    Given a ranked preference profile and a set of candidate positions, 
    generates a set of voter positions that are consistent with each voters ranking.
    
    Args:
        voter_dist_fn (Callable[..., np.ndarray], optional): Distribution to sample a single
            voter's position from, defaults to uniform distribution.
        voter_dist_fn_params: (dict[str, Any], optional): Parameters to be passed to
            voter_dist_fn, defaults to None,
            which creates the unif(0,1) distribution in 2 dimensions.
        distance_fn: (Callable[[np.ndarray, np.ndarray], float]], optional):
            Computes distance between a voter and a candidate,
            defaults to euclidean distance.

    Attributes:
        voter_dist_fn (Callable[..., np.ndarray], optional): Distribution to sample a single
            voter's position from, defaults to uniform distribution.
        voter_dist_fn_params: (dict[str, Any], optional): Parameters to be passed to
            voter_dist_fn, defaults to None, 
            which creates the unif(0,1) distribution in 2 dimensions.
        distance_fn: (Callable[[np.ndarray, np.ndarray], float]], optional):
            Computes distance between a voter and a candidate,
            defaults to euclidean distance.
    """
    def __init__(
        self,
        voter_dist_fn: Callable = np.random.uniform,
        voter_dist_fn_params: Dict[str, Any] = None,
        distance_fn: Callable =euclidean_distance
    ):
        # See note above.
        warnings.warn("This spatial generation method has not been thoroughly tested.")
        
        self.voter_dist_fn = voter_dist_fn

        if voter_dist_fn_params is None:
            if voter_dist_fn is np.random.uniform:
                self.voter_dist_fn_params = {"low": 0.0, "high": 1.0, "size": 2.0}
            else:
                raise ValueError(
                    "No parameters were given for the input voter distribution."
                )
        else:
            try:
                self.voter_dist_fn(**voter_dist_fn_params)
            except TypeError:
                raise TypeError("Invalid parameters for the voter distribution.")

            self.voter_dist_fn_params = voter_dist_fn_params

        try:
            v = self.voter_dist_fn(**self.voter_dist_fn_params)
            c = self.voter_dist_fn(**self.voter_dist_fn_params)
            distance_fn(v, c)
        except TypeError:
            raise ValueError(
                "Distance function is invalid or incompatible "
                "with voter/candidate distributions."
            )

        self.distance_fn = distance_fn
        
    def uniform_ranking_sample(
        self,
        ranking : NDArray,
        candidate_positions : NDArray
    ) -> NDArray:
        """
        Given a partial ranking of candidates, generates a voter position
        that is consistent with the ranking.
        
        Args:
            ranking (NDArray): A partial ranking of candidates.
            candidate_positions (NDArray): A m x d matrix where each row i contains the
                position of candidate i in metric space.
                
        Returns:
            sample (NDArray): A d-dimensional vector representing a voter's position.
        """
        sample = None
        found = False
        iters = 0
        while not found:
            iters += 1
            x = self.voter_dist_fn(**self.voter_dist_fn_params)
            cand_distances = np.linalg.norm(candidate_positions - x, axis = 1)
            distance_ranking = tiebreak(cand_distances)
            if np.array_equal(ranking, distance_ranking[:len(ranking)]):
                sample = x
                found = True
            if iters > 1000000:
                raise ValueError(
                    'Could not find valid position within 1 million samples, '
                    'try increasing the dimension.'
                )
        return sample


    def generate(
        self,
        n : int,
        m : int,
        profile : NDArray,
        candidate_positions : NDArray,
        candidate_labels : Optional[NDArray] = None,
        voter_labels : Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Samples a metric position for n voters from
        the voter distribution. Samples a metric position for each candidate
        from the input candidate distribution. With sampled
        positions, this method then creates a ranked preference profile in which
        voter's preferences are consistent with their distances to the candidates
        in the metric space.

        Args:
            profile (NDArray): A m x n preference profile with each column j containing
                voter j's rankings for each of the m candidates.
            candidate_positions (NDArray): A m x d matrix where each row i contains the
                position of candidate i in metric space.
            candidate_labels (NDArray): A length n array where each element i contains the
                group label for candidate i. Defaults to None, in which case distinct labels 
                are given to each candidate.
            voter_labels (NDArray): A length n array where each element i contains the
                group label for voter i. Defaults to None, in which case voter labels 
                are assigned as the group label of the candidate they rank first.
            
        Returns:
            profile (NDArray): A m x n preference profile with each column j containing 
                voter j's rankings for each of the m candidates.
            candidate_positions (NDArray): A m x d matrix where each row i contains the
                position of candidate i in metric space. 
            voter_positions (NDArray): A n x d matrix where each row i contains the
                position of voter i in metric space.
            candidate_labels (NDArray): A length m array where each element i contains the
                group label for voter i.
            voter_labels (NDArray): A length n array where each element i contains the
                group label for voter i.
        """
        if profile.shape[0] != m:
            raise ValueError(
                "Number of candidates in profile does not match number of candidates given."
            )
            
        if profile.shape[1] != n:
            raise ValueError(
                "Number of voters in profile does not match number of voters given."
            )
            
        if candidate_positions.shape[0] != m:
            raise ValueError(
                "Number of candidates in the position array "
                "does not match number of candidates given."
            )
            
        _,d = candidate_positions.shape
        
        voter_positions = np.zeros((n,d))
        for i in range(n):
            ranking = profile[:,i]
            ranking = ranking[ranking != -1]
            voter_positions[i,:] = self.uniform_ranking_sample(ranking, candidate_positions)

        if candidate_labels is None:
            candidate_labels = np.arange(m)
        
        if voter_labels is None:
            voter_labels = candidate_labels[profile[0,:]]
        
        # Recompute Preference Profile (Since the original may have been incomplete)
        cst_array = cost_array(voter_positions, candidate_positions, self.distance_fn)
        complete_profile = cost_array_to_ranking(cst_array)

        return (
            complete_profile,
            candidate_positions,
            voter_positions,
            candidate_labels,
            voter_labels
        )