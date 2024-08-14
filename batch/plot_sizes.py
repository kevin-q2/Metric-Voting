import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.append(os.path.join(os.getcwd(), 'metric_voting/code'))
from elections import SNTV,Bloc,STV,Borda, ChamberlinCourant, Monroe, GreedyCC, PluralityVeto, SMRD, OMRD, DMRD
from tools import group_representation, max_group_representation


# Specify results to plot from:
input_file = 'metric_voting/data/2party.npz'

# And where to save them!
output_file1 = 'metric_voting/figures/2party_sizes.png'
output_file2 = 'metric_voting/figures/2party_sizes_overall.png'


# Read data
loaded_data = np.load(input_file)
result_dict = {key: loaded_data[key] for key in loaded_data.files}


# Specify elections used (and number of samples for each)
elections_list = [SNTV, Bloc, STV, Borda, ChamberlinCourant, GreedyCC, Monroe, PluralityVeto, SMRD, OMRD, DMRD]
n_samples = 1000


# Specify global parameters for matplotlib
pal = sns.color_palette("hls", 8)
tab20_colors = plt.cm.tab20.colors

#plt.rcParams['font.family'] = 'serif'        # Use a serif font
#plt.rcParams['font.serif'] = ['Times New Roman']  # Specify the font family
plt.rcParams['font.size'] = 16               # Set the default font size

####################################################################################################################################
# Compute results

group_sizes = [[100 - i, i] for i in range(0, 105, 5)]
group_select = 1
num_sizes = len(group_sizes)

size_avg_represent = {e.__name__:(np.zeros(num_sizes), np.zeros(num_sizes)) for e in elections_list}
size_avg_represent1 = {e.__name__:(np.zeros(num_sizes), np.zeros(num_sizes)) for e in elections_list}

for s in range(num_sizes):
    f = 'metric_voting/data/2sizes' + str(s) + '.npz'
    loaded_data = np.load(f)
    result_dict = {key: loaded_data[key] for key in loaded_data.files}
    
    s_avg_represent = {e.__name__:np.zeros(n_samples) for e in elections_list}
    s_avg_represent1 = {e.__name__:np.zeros(n_samples) for e in elections_list}
    
    for i in range(n_samples):
        voter_positions = result_dict['voters'][i]
        candidate_positions = result_dict['candidates'][i]
        labels = result_dict['labels'][i]
        labels1 = np.zeros(len(labels))

        for j,E in enumerate(elections_list):
            name = E.__name__
                
            winners = result_dict[name][i]
            represent = group_representation(voter_positions, candidate_positions, labels, winners, group_select, size = None)
            represent1 = group_representation(voter_positions, candidate_positions, labels1, winners, 0, size = None)
            s_avg_represent[name][i] = represent
            s_avg_represent1[name][i] = represent1
            
    for ename, evals in s_avg_represent.items():
        size_avg_represent[ename][0][s] = np.mean(evals)
        size_avg_represent[ename][1][s] = np.std(evals)
        
    for ename, evals in s_avg_represent1.items():
        size_avg_represent1[ename][0][s] = np.mean(evals)
        size_avg_represent1[ename][1][s] = np.std(evals)
        
##############################################################################################################
# Plot results

fig,ax = plt.subplots(figsize=(10, 6), dpi = 200)

Asizes = [x[group_select]/100 for x in group_sizes]
for i, (ename,evals) in enumerate(size_avg_represent.items()):
    ax.plot(Asizes, evals[0], label=ename, color = tab20_colors[i], linewidth = 3, marker = 'o')
    #ax.fill_between(Asizes, evals[0] - evals[1], evals[0] + evals[1], color=tab20_colors[i], alpha=0.05)

#plt.ylabel(r'$\alpha$')
#plt.xlabel('Bloc size')
plt.legend(fontsize = 10, loc = 'upper left')
plt.savefig(output_file1, bbox_inches='tight')
plt.show()


###############################################################################################################

fig,ax = plt.subplots(figsize=(10, 6), dpi = 200)

Asizes = [x[group_select]/100 for x in group_sizes]
for i, (ename,evals) in enumerate(size_avg_represent1.items()):
    ax.plot(Asizes, evals[0], label=ename, color = tab20_colors[i], linewidth = 3, marker = 'o')
    #ax.fill_between(Asizes, evals[0] - evals[1], evals[0] + evals[1], color=tab20_colors[i], alpha=0.05)

#plt.ylabel(r'$\alpha$')
#plt.xlabel('Bloc size')
plt.legend(fontsize = 10, loc = 'upper left')
plt.savefig(output_file2, bbox_inches='tight')
plt.show()