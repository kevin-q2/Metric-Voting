import numpy as np
from metric_voting.elections import *
from metric_voting.measurements import *
from metric_voting.utils import *


np.random.seed(918717)
input_file = 'data/two_bloc/thousand_voters/samples.npz'
output_file = 'data/two_bloc/thousand_voters/samples_with_pav.npz'

loaded_data = np.load(input_file)
result_dict = {key: loaded_data[key] for key in loaded_data.files}


n = 1000
m = 20
k = 4
voters = result_dict['voters']
candidates = result_dict['candidates']
samples = len(voters)
E = PAV(solver = 'GUROBI_CMD', log_path = 'experiments/two_bloc/thousand_voters/pav2.log')
pav_results = np.zeros((samples, m))
for i in range(samples):
    voter_positions = voters[i]
    candidate_positions = candidates[i]
    cst_array = euclidean_cost_array(voter_positions, candidate_positions)
    profile = cost_array_to_ranking(cst_array)
    winners = E.elect(profile, k)
    pav_results[i,winners] = True
    
result_dict['PAV'] = pav_results
np.savez_compressed(output_file, **result_dict)