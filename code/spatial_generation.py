import numpy as np
from tools import euclidean_distance


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
    def __init__(self, m, voter_dist = np.random.uniform, voter_params = None, 
                 candidate_dist = np.random.uniform, candidate_params = None, 
                 distance = euclidean_distance):
        
        self.m = m
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



    def generate(self, n):
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
            [self.candidate_dist(**self.candidate_params) for c in range(self.m)]
        )
        voter_positions = np.array(
            [self.voter_dist(**self.voter_params) for v in range(n)]
        )
        
        voter_labels = np.zeros(n)
        
        profile = np.zeros((self.m,n), dtype = np.int64)
        for i in range(n):
            distances = [self.distance(voter_positions[i,:], candidate_positions[j,:]) 
                         for j in range(self.m)]
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
    previous by taking different distributions for each group of voters. 
    
    Args:
    m (int): Number of candidates.
    g (int): Number of groups.
    voter_dists (list[Callable[..., np.ndarray]], optional): Distribution to sample a single
        voter's position from, defaults to uniform distribution.
    voter_params: (list[dict[str, Any]], optional): Parameters to be passed to
        voter_dists, defaults to None, which creates the unif(0,1) distribution in 2 dimensions.
    candidate_dist: (Callable[..., np.ndarray], optional): Distribution to sample a
        single candidate's position from, defaults to uniform distribution.
    candidate_params: (Optional[Dict[str, Any]], optional): Parameters to be passed
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
    candidate_dist: (Callable[..., np.ndarray], optional): Distribution to sample a
        single candidate's position from, defaults to uniform distribution.
    candidate_params: (Optional[Dict[str, Any]], optional): Parameters to be passed
        to candidate_dist, defaults to None, which creates the unif(0,1)
        distribution in 2 dimensions.
    distance: (Callable[[np.ndarray, np.ndarray], float]], optional):
        Computes distance between a voter and a candidate,
        defaults to euclidean distance.
    """
    def __init__(self, m, g, voter_dists = None, voter_params = None, 
                 candidate_dist = None, candidate_params = None, 
                 distance = euclidean_distance):
        
        self.m = m
        self.g = g
        
        if voter_dists is None:
            self.voter_dists = [np.random.uniform]*g
        else:
            if len(voter_dists) != g:
                raise ValueError('Group size does not match with given voter distributions')
             
            self.voter_dists = voter_dists
        
        self.candidate_dist = candidate_dist
        
        
        if voter_params is None:
            self.voter_params = [{} for _ in range(g)]
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
            v = self.voter_dists[0](**self.voter_params[0])
            c = self.candidate_dist(**self.candidate_params)
            distance(v, c)
        except TypeError:
            raise ValueError(
                "Distance function is invalid or incompatible "
                "with voter/candidate distributions."
            )

        self.distance = distance



    def generate(self, G):
        """ 
        Samples a metric position for n voters from
        the voter distribution. Samples a metric position for each candidate
        from the input candidate distribution. With sampled
        positions, this method then creates a ranked preference profile in which
        voter's preferences are consistent with their distances to the candidates
        in the metric space.
        
        Args:
            G (list[int]): List describing the number of voters per group.
        """
        
        if len(G) != self.g:
            raise ValueError('Groups of voters and distributions do not match')
        
        n = np.sum(G)
        
        candidate_positions = np.array(
            [self.candidate_dist(**self.candidate_params) for c in range(self.m)]
        )
        
        voter_positions = [[self.voter_dists[i](**self.voter_params[i]) for v in range(G[i])] for i in range(self.g)]
        voter_positions = np.vstack(voter_positions)
        
        voter_labels = [[i]*G[i] for i in range(self.g)]
        voter_labels = np.array([item for sublist in voter_labels for item in sublist])
        voter_labels = voter_labels.flatten()
        
        profile = np.zeros((self.m,n), dtype = np.int64)
        for i in range(n):
            distances = [self.distance(voter_positions[i,:], candidate_positions[j,:]) 
                         for j in range(self.m)]
            ranking = np.argsort(distances)
            profile[:,i] = ranking
            
        return profile, candidate_positions, voter_positions, voter_labels