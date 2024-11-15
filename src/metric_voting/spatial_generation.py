import numpy as np
from .utils import euclidean_distance


class Spatial:
    """
    Spatial model for ballot generation. In some metric space determined
    by an input distance function, randomly sample each voter's and
    each candidate's positions from input voter and candidate distributions.
    Using generate_profile() outputs a ranked profile which is consistent
    with the sampled positions (respects distances).
    
    Args:
    m (int): Number of candidates.
    voter_dist (Callable[..., np.ndarray], optional): Distribution to sample a single
        voter's position from, defaults to uniform distribution.
    voter_params: (dict[str, Any], optional): Parameters to be passed to
        voter_dist, defaults to None, which creates the unif(0,1) distribution in 2 dimensions.
    candidate_dist: (Callable[..., np.ndarray], optional): Distribution to sample a
        single candidate's position from, defaults to uniform distribution.
    candidate_params: (dict[str, Any], optional): Parameters to be passed
        to candidate_dist, defaults to None, which creates the unif(0,1)
        distribution in 2 dimensions.
    distance: (Callable[[np.ndarray, np.ndarray], float]], optional):
        Computes distance between a voter and a candidate,
        defaults to euclidean distance.
        
    Attributes:
    m (int): Number of candidates.
    voter_dist (Callable[..., np.ndarray], optional): Distribution to sample a single
        voter's position from, defaults to uniform distribution.
    voter_params: (dict[str, Any], optional): Parameters to be passed to
        voter_dist, defaults to None, which creates the unif(0,1) distribution in 2 dimensions.
    candidate_dist: (Callable[..., np.ndarray], optional): Distribution to sample a
        single candidate's position from, defaults to uniform distribution.
    candidate_params: (dict[str, Any], optional): Parameters to be passed
        to candidate_dist, defaults to None, which creates the unif(0,1)
        distribution in 2 dimensions.
    distance: (Callable[[np.ndarray, np.ndarray], float]], optional):
        Computes distance between a voter and a candidate,
        defaults to euclidean distance.
    """
    def __init__(self, voter_dist = np.random.uniform, voter_params = None, 
                 candidate_dist = np.random.uniform, candidate_params = None, 
                 distance = euclidean_distance):
        
        self.voter_dist = voter_dist
        self.candidate_dist = candidate_dist
        
        if voter_params is None:
            if voter_dist is np.random.uniform:
                self.voter_params = {"low": 0.0, "high": 1.0, "size": 2.0}
            else:
                raise ValueError(
                    "No parameters were given for the input voter distribution."
                )
        else:
            try:
                self.voter_dist(**voter_params)
            except TypeError:
                raise TypeError("Invalid parameters for the voter distribution.")

            self.voter_params = voter_params

        if candidate_params is None:
            if candidate_dist is np.random.uniform:
                self.candidate_params = {"low": 0.0, "high": 1.0, "size": 2.0}
            else:
                raise ValueError(
                    "No parameters were given for the input candidate distribution."
                )
        else:
            try:
                self.candidate_dist(**candidate_params)
            except TypeError:
                raise TypeError("Invalid parameters for the candidate distribution.")

            self.candidate_params = candidate_params

        try:
            v = self.voter_dist(**self.voter_params)
            c = self.candidate_dist(**self.candidate_params)
            distance(v, c)
        except TypeError:
            raise ValueError(
                "Distance function is invalid or incompatible "
                "with voter/candidate distributions."
            )

        self.distance = distance



    def generate(self, n, m):
        """ 
        Samples a metric position for n voters from
        the voter distribution. Samples a metric position for each candidate
        from the input candidate distribution. With sampled
        positions, this method then creates a ranked preference profile in which
        voter's preferences are consistent with their distances to the candidates
        in the metric space.
        
        Args:
            n (int): _Number of voters. 
        """
        
        candidate_positions = np.array(
            [self.candidate_dist(**self.candidate_params) for c in range(m)]
        )
        voter_positions = np.array(
            [self.voter_dist(**self.voter_params) for v in range(n)]
        )
        
        voter_labels = np.zeros(n)
        
        profile = np.zeros((m,n), dtype = np.int64)
        for i in range(n):
            distances = [self.distance(voter_positions[i,:], candidate_positions[j,:]) 
                         for j in range(m)]
            ranking = np.argsort(distances)
            profile[:,i] = ranking
            
        return profile, candidate_positions, voter_positions, voter_labels
    
    
    
    

class GroupSpatial:
    """
    Spatial model for ballot generation. In some metric space determined
    by an input distance function, randomly sample each voter's and
    each candidate's positions from input voter and candidate distributions.
    Using generate_profile() outputs a ranked profile which is consistent
    with the sampled positions (respects distances). This model differs from the
    previous by allowing for different distributions for different blocs of voters,
    as well as different distributions for groups of candidates. 
    
    Args:
    m (int): Number of candidates.
    g (int): Number of groups.
    voter_dists (list[Callable[..., np.ndarray]], optional): Distributions to sample
        voter's positions from, defaults to uniform distribution.
    voter_params: (list[dict[str, Any]], optional): Parameters to be passed to
        voter_dists, defaults to None, which creates the unif(0,1) distribution in 2 dimensions.
    candidate_dists (list[Callable[..., np.ndarray]], optional): Distributions to sample
        candidate's positions from, defaults to uniform distribution.
    candidate_params (list[dict[str, Any]], optional): Parameters to be passed
        to candidate_dist, defaults to None, which creates the unif(0,1)
        distribution in 2 dimensions.
    distance: (Callable[[np.ndarray, np.ndarray], float]], optional):
        Computes distance between a voter and a candidate,
        defaults to euclidean distance.
        
    Attributes:
    m (int): Number of candidates.
    g (int): Number of groups.
    voter_dists (list[Callable[..., np.ndarray]], optional): Distribution to sample a single
        voter's position from, defaults to uniform distribution.
    voter_params: (list[dict[str, Any]], optional): Parameters to be passed to
        voter_dists, defaults to None, which creates the unif(0,1) distribution in 2 dimensions.
    candidate_dists: (Callable[..., np.ndarray], optional): Distribution to sample a
        single candidate's position from, defaults to uniform distribution.
    candidate_params: (Optional[Dict[str, Any]], optional): Parameters to be passed
        to candidate_dist, defaults to None, which creates the unif(0,1)
        distribution in 2 dimensions.
    distance: (Callable[[np.ndarray, np.ndarray], float]], optional):
        Computes distance between a voter and a candidate,
        defaults to euclidean distance.
    """
    def __init__(self, voter_groups, candidate_groups,
                 voter_dists = None, voter_params = None, 
                 candidate_dists = None, candidate_params = None, 
                 distance = euclidean_distance):
        
        self.voter_groups = voter_groups
        self.candidate_groups = candidate_groups
        
        if voter_dists is None:
            self.voter_dists = [np.random.uniform]*voter_groups
        else:
            if len(voter_dists) != voter_groups:
                raise ValueError('Group size does not match with given voter distributions')
             
            self.voter_dists = voter_dists
            
        if candidate_dists is None:
            self.candidate_dists = [np.random.uniform]*candidate_groups
        else:
            if len(candidate_dists) != candidate_groups:
                raise ValueError('Group size does not match with given candidate distributions')
             
            self.candidate_dists = candidate_dists
        
        if voter_params is None:
            self.voter_params = [{} for _ in range(voter_groups)]
            for i,dist in enumerate(self.voter_dists):
                if dist is np.random.uniform:
                    self.voter_params[i] = {"low": 0.0, "high": 1.0, "size": 2.0}
                else:
                    raise ValueError(
                        "No parameters were given for the input voter distribution."
                    )
        else:
            for i,dist in enumerate(self.voter_dists):
                try:
                    dist(**voter_params[i])
                except TypeError:
                    raise TypeError("Invalid parameters for the voter distribution.")

            self.voter_params = voter_params
            
            
        if candidate_params is None:
            self.candidate_params = [{} for _ in range(candidate_groups)]
            for i,dist in enumerate(self.candidate_dists):
                if dist is np.random.uniform:
                    self.candidate_params[i] = {"low": 0.0, "high": 1.0, "size": 2.0}
                else:
                    raise ValueError(
                        "No parameters were given for the input voter distribution."
                    )
        else:
            for i,dist in enumerate(self.candidate_dists):
                try:
                    dist(**candidate_params[i])
                except TypeError:
                    raise TypeError("Invalid parameters for the voter distribution.")

            self.candidate_params = candidate_params

        try:
            v = self.voter_dists[0](**self.voter_params[0])
            c = self.candidate_dists[0](**self.candidate_params[0])
            distance(v, c)
        except TypeError:
            raise ValueError(
                "Distance function is invalid or incompatible "
                "with voter/candidate distributions."
            )

        self.distance = distance



    def generate(self, n, m, voter_size_dist, candidate_size_dist, exact = True):
        """ 
        Samples a metric position for n voters from
        the voter distribution. Samples a metric position for each candidate
        from the input candidate distribution. With sampled
        positions, this method then creates a ranked preference profile in which
        voter's preferences are consistent with their distances to the candidates
        in the metric space.
        
        Args:
            n (int): Number of voters.
            voter_group_dist (list[float]): List describing the distribution of sizes for voter
                groups. If exact = True these are taken to be the exact sizes of the groups. Otherwise
                they are taken to be the relative probability of belonging to each of the groups.
            candidate_group_dist (list[float]): List describing the distribution of sizes for candidate
                groups. If exact = True these are taken to be the exact sizes of the groups. Otherwise
                they are taken to be the relative probability of belonging to each of the groups.
        """
        
        if len(voter_size_dist) != self.voter_groups:
            raise ValueError('Groups of voters and distributions do not match')
        
        elif len(candidate_size_dist) != self.candidate_groups:
            raise ValueError('Groups of candidates and distributions do not match')
        
        elif exact and (sum(voter_size_dist) != n or sum(candidate_size_dist) != m):
            raise ValueError('Number of voters or candidates do not match')
        
        elif not exact and (sum(voter_size_dist) != 1 or sum(candidate_size_dist) != 1):
            raise ValueError('Distribution sizes do not sum to 1')
         
         
        if exact:
            voter_sizes = voter_size_dist
            candidate_sizes = candidate_size_dist  
        else:
            voter_groups = np.random.choice(self.voter_groups, n, p = voter_size_dist)
            voter_sizes = [np.sum(voter_groups == i) for i in range(self.voter_groups)]
            
            candidate_groups = np.random.choice(self.candidate_groups, m, p = candidate_size_dist)
            candidate_sizes = [np.sum(candidate_groups == i) for i in range(self.candidate_groups)]
        
        
        #candidate_positions = np.array(
        #    [self.candidate_dist(**self.candidate_params) for c in range(self.m)]
        #)
        
        candidate_positions = [[self.candidate_dists[i](**self.candidate_params[i]) for _ in range(candidate_sizes[i])] 
                           for i in range(self.candidate_groups) if candidate_sizes[i] != 0]
        candidate_positions = np.vstack(candidate_positions)
        
        voter_positions = [[self.voter_dists[i](**self.voter_params[i]) for _ in range(voter_sizes[i])] 
                           for i in range(self.voter_groups) if voter_sizes[i] != 0]
        voter_positions = np.vstack(voter_positions)
        
        voter_labels = [[i]*voter_sizes[i] for i in range(self.voter_groups)]
        voter_labels = np.array([item for sublist in voter_labels for item in sublist])
        voter_labels = voter_labels.flatten()
        
        profile = np.zeros((m,n), dtype = np.int64)
        for i in range(n):
            distances = [self.distance(voter_positions[i,:], candidate_positions[j,:]) 
                         for j in range(m)]
            ranking = np.argsort(distances)
            profile[:,i] = ranking
            
        return profile, candidate_positions, voter_positions, voter_labels