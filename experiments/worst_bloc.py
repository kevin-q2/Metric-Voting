import numpy as np
from metric_voting.measurements import euclidean_cost_array, worst_random_group_inefficiency


np.random.seed(918717)
input_file = 'data/2bloc_new.npz'
output_file = 'data/2bloc_worst.npz'

loaded_data = np.load(input_file)
result_dict = {key: loaded_data[key] for key in loaded_data.files}
voters = result_dict['voters']
candidates = result_dict['candidates']
labels = result_dict['labels']

worst_bloc_dict = {
    'voters' : voters,
    'candidates' : candidates,
    'labels' : labels
}

elections = [_ for _ in result_dict.keys() if _ not in ['voters', 'candidates', 'labels']]
n_samples = result_dict[elections[0]].shape[0]
samples_per = 1000

for e in elections:
    worst_bloc_samples = []
    for j in range(n_samples):
        V = voters[j]
        C = candidates[j]
        cst_array = euclidean_cost_array(V,C)
        winner_indices = np.where(result_dict[e][j])[0]
        worst_score, worst_bloc = worst_random_group_inefficiency(
            samples_per,
            cst_array,
            winner_indices
        )
        worst_mask = np.zeros(V.shape[0], dtype = bool)
        worst_mask[worst_bloc] = True
        worst_bloc_samples.append(worst_mask)
        
    worst_bloc_dict[e] = np.array(worst_bloc_samples)
    
    
np.savez_compressed(output_file, **worst_bloc_dict)