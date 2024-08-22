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
from elections import SNTV,Bloc,STV,Borda, ChamberlinCourant, Monroe, GreedyCC, PluralityVeto, SMRD, OMRD, DMRD, ExpandingApprovals
from tools import group_representation, max_group_representation

# Specify results to plot from:
input_file = 'metric_voting/data/2bloc.npz'

# And where to save them!
output_file1 = 'metric_voting/figures/2bloc_representation.png'
output_file2 = 'metric_voting/figures/2bloc_representation_overall.png'


# Read data
loaded_data = np.load(input_file)
result_dict = {key: loaded_data[key] for key in loaded_data.files}


# Specify elections used (and number of samples for each)
elections_list = [SNTV, Bloc, STV, Borda, ChamberlinCourant, GreedyCC, Monroe, PluralityVeto, ExpandingApprovals, SMRD, OMRD, DMRD]
#elections_list = [SNTV, Bloc, STV, Borda, GreedyCC, PluralityVeto, ExpandingApprovals, SMRD, OMRD, DMRD]
n_samples = 10000


# Specify global parameters for matplotlib
colors = ["#0099cd","#ffca5d","#00cd99","#99cd00","#cd0099","#9900cd","#8dd3c7",
        "#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5","#bc80bd",
        "#ccebc5","#ffed6f","#ffffb3","#a6cee3","#1f78b4","#b2df8a","#33a02c",
        "#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a","#b15928",
        "#64ffda","#00B8D4","#A1887F","#76FF03","#DCE775","#B388FF","#FF80AB",
        "#D81B60","#26A69A","#FFEA00","#6200EA",
    ]

colors = colors[:6] + colors[-12::2]

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.serif": [],
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 24
})




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
        
represent_data = [values[~np.isnan(values)] for values in avg_represent.values()]
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

represent_labels = ['CC' if n == 'ChamberlinCourant' else n for n in represent_labels]
represent_labels = ['Expanding' if n == 'ExpandingApprovals' else n for n in represent_labels]
represent_labels_overall = ['CC' if n == 'ChamberlinCourant' else n for n in represent_labels_overall]
represent_labels_overall = ['Expanding' if n == 'ExpandingApprovals' else n for n in represent_labels_overall]


######################################################################################################################
# Plot first box
plt.figure(figsize=(16, 6), dpi = 200)
flierprops = dict(marker='o', markerfacecolor='none', markersize=2, linestyle='none', markeredgecolor='black', alpha = 0.5)

# BUG FIX for now:
represent_data = np.array(represent_data)
mask = np.all(represent_data >= 1, axis=0)
filtered_data = represent_data[:, mask]

ax = sns.boxplot(data=filtered_data.T, palette = colors, width = 0.6, linewidth=2.5, fliersize= 1, flierprops=flierprops)
ax.set_ylim(0.9, 2)
plt.xticks(ticks=np.arange(len(elections_list)), labels=represent_labels, rotation=67, ha='right')
plt.ylabel('group-inefficiency')
plt.savefig(output_file1, bbox_inches='tight')

########################################################################################################################
# Plot second violin
plt.figure(figsize=(16, 6), dpi = 200)
flierprops = dict(marker='o', markerfacecolor='none', markersize=2, linestyle='none', markeredgecolor='black', alpha = 0.5)

# BUG FIX for now:
represent_data_overall = np.array(represent_data_overall)
mask = np.all(represent_data_overall >= 1, axis=0)
filtered_data_overall = represent_data_overall[:, mask]

ax = sns.boxplot(data=filtered_data_overall.T, palette = colors, width = 0.6, linewidth=2.5, fliersize= 1, flierprops=flierprops)
ax.set_ylim(0.9, 2)
plt.xticks(ticks=np.arange(len(elections_list)), labels=represent_labels_overall, rotation=67, ha='right')
plt.ylabel('overall-inefficiency')
plt.savefig(output_file2, bbox_inches='tight')


#########################################################################################################################

plt.figure(figsize=(16, 6), dpi = 200)

# BUG FIX for now:
represent_data = np.array(represent_data)
mask = np.all(represent_data >= 1, axis=0)
filtered_data = represent_data[:, mask]
#random_columns = np.random.choice(filtered_data.shape[1], 2500, replace=False)

ax = sns.violinplot(data=filtered_data.T, palette = colors, alpha = 1, width = 0.9)
for violin in ax.collections:
    violin.set_alpha(1)
    
ax.set_ylim(0.5, 2)
plt.xticks(ticks=np.arange(len(elections_list)), labels=represent_labels, rotation=67)
plt.ylabel('group-inefficiency')
plt.savefig(output_file1[:-4] + '_violin' + output_file1[-4:], bbox_inches='tight')


#####################################################################################################################

plt.figure(figsize=(16, 6), dpi = 200)

# BUG FIX for now:
represent_data_overall = np.array(represent_data_overall)
mask = np.all(represent_data_overall >= 1, axis=0)
filtered_data_overall = represent_data_overall[:, mask]
#random_columns = np.random.choice(filtered_data_overall.shape[1], 2500, replace=False)

ax = sns.violinplot(data=filtered_data_overall.T, palette = colors, alpha = 1, width = 0.9)
for violin in ax.collections:
    violin.set_alpha(1)
    
ax.set_ylim(0.5, 2)
plt.xticks(ticks=np.arange(len(elections_list)), labels=represent_labels_overall, rotation=67)
plt.ylabel('overall-inefficiency')
plt.savefig(output_file2[:-4] + '_violin' + output_file2[-4:], bbox_inches='tight')

#################################################################################################33