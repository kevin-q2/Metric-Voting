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
input_file = 'metric_voting/data/2party_10k.npz'

# And where to save them!
output_file1 = 'metric_voting/figures/2party_representation_10k.png'
output_file2 = 'metric_voting/figures/2party_representation_10k_overall.png'


# Read data
loaded_data = np.load(input_file)
result_dict = {key: loaded_data[key] for key in loaded_data.files}


# Specify elections used (and number of samples for each)
#elections_list = [SNTV, Bloc, STV, Borda, ChamberlinCourant, GreedyCC, Monroe, PluralityVeto, SMRD, OMRD, DMRD]
elections_list = [SNTV, Bloc, STV, Borda, GreedyCC, PluralityVeto, SMRD, OMRD, DMRD]
n_samples = 10000


# Specify global parameters for matplotlib
pal = sns.color_palette("hls", 8)
tab20_colors = plt.cm.tab20.colors

#plt.rcParams['font.family'] = 'serif'        # Use a serif font
#plt.rcParams['font.serif'] = ['Times New Roman']  # Specify the font family
plt.rcParams['font.size'] = 16               # Set the default font size




##############################################################################################################
# Compute results:
# 1) Using known group labels
avg_represent = {e.__name__:np.zeros(n_samples) for e in elections_list}

group_select = 0
for i in range(n_samples):
    voter_positions = result_dict['voters'][i]
    candidate_positions = result_dict['candidates'][i]
    labels = result_dict['labels'][i]
    
    for j,E in enumerate(elections_list):
        name = E.__name__
        winners = result_dict[name][i]
        represent = group_representation(voter_positions, candidate_positions, labels, winners, group_select, size = None)
        avg_represent[name][i] = represent
        
represent_data = [values for values in avg_represent.values()]
represent_labels = [name for name in avg_represent.keys()]

# 2) Treat everyone as one group
avg_represent_overall = {e.__name__:np.zeros(n_samples) for e in elections_list}

group_select = 0
for i in range(n_samples):
    voter_positions = result_dict['voters'][i]
    candidate_positions = result_dict['candidates'][i]
    labels = result_dict['labels'][i]
    labels = np.zeros(len(labels))
    
    for j,E in enumerate(elections_list):
        name = E.__name__
        winners = result_dict[name][i]
        represent = group_representation(voter_positions, candidate_positions, labels, winners, group_select, size = None)
        avg_represent_overall[name][i] = represent
        
represent_data_overall = [values for values in avg_represent_overall.values()]
represent_labels_overall = [name for name in avg_represent_overall.keys()]

ylimit = max(max([np.max(r) for r in represent_data]), max([np.max(r) for r in represent_data_overall]))


######################################################################################################################
# Plot first violin
plt.figure(figsize=(16, 6), dpi = 200)
ax = sns.violinplot(data=represent_data, palette = tab20_colors, alpha = 1, width = 0.9)
for violin in ax.collections:
    violin.set_alpha(1)
    
ax.set_ylim(0.9, 2)
# NOTE: Uncomment if you want to see the labels!
#plt.xticks(ticks=np.arange(len(elections_list)), labels=represent_labels, rotation=67)
#plt.ylabel(r'$\alpha$')
plt.savefig(output_file1, bbox_inches='tight')

########################################################################################################################
# Plot second violin

plt.rcParams.update({'font.size': 18})
# Create a violin plot
plt.figure(figsize=(16, 6), dpi = 200)
ax = sns.violinplot(data=represent_data_overall, palette = tab20_colors, alpha = 1, width = 0.9)
for violin in ax.collections:
    violin.set_alpha(1)
    
ax.set_ylim(0.9, 2)
# NOTE: Uncomment if you want to see the labels!
#plt.xticks(ticks=np.arange(len(elections_list)), labels=represent_labels1, rotation=67)
#plt.ylabel(r'$\alpha$')
plt.savefig(output_file2, bbox_inches='tight')


#########################################################################################################################
