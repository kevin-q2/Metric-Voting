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
input_file = 'metric_voting/data/4party.npz'

# And where to save them!
output_file = 'metric_voting/figures/4party.png'


# Read data
loaded_data = np.load(input_file)
result_dict = {key: loaded_data[key] for key in loaded_data.files}


# Specify elections used
elections_list = [SNTV, Bloc, STV, Borda, ChamberlinCourant, GreedyCC, Monroe, PluralityVeto, SMRD, OMRD, DMRD]


# Specify global parameters for matplotlib
pal = sns.color_palette("hls", 8)
tab20_colors = plt.cm.tab20.colors

#plt.rcParams['font.family'] = 'serif'        # Use a serif font
#plt.rcParams['font.serif'] = ['Times New Roman']  # Specify the font family
plt.rcParams['font.size'] = 16               # Set the default font size



###########################################################################################################
# Plotting:

fig, axes = plt.subplots(len(elections_list) + 1, 3, figsize=(10, 24), dpi = 200)

for i, ax in enumerate(axes.flat):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')

#vc = pal[5]
#cc = pal[4]
#wc = pal[7]

vc = tab20_colors[1]
cc = tab20_colors[5]
wc = tab20_colors[8]

# Index for the sample used for the example plots:
sample_idx = 1
voter_example = result_dict['voters'][sample_idx]
voter_stack = pd.DataFrame(np.vstack(result_dict['voters']), columns = ['x','y'])
# Use a smaller sample for KDE plots (otherwise takes a while to run without much added benefit)
voter_stack_sample = voter_stack.sample(frac=0.25, random_state=42)

# Just showing the distribution without the election
candidate_example = result_dict['candidates'][sample_idx]
candidate_stack = pd.DataFrame(np.vstack(result_dict['candidates']), columns = ['x','y'])
# Use a smaller sample for KDE plots (otherwise takes a while to run without much added benefit)
candidate_stack_sample = candidate_stack.sample(frac=0.25, random_state=42)

# Set x and y limits for scatter and example plots:
epsilon = 0.5
scatter_xlim = [min(voter_stack.iloc[:,0].min(), candidate_stack.iloc[:,0].min()) - epsilon,
                max(voter_stack.iloc[:,0].max(), candidate_stack.iloc[:,0].max()) + epsilon]
scatter_ylim = [min(voter_stack.iloc[:,1].min(), candidate_stack.iloc[:,1].min()) - epsilon,
                max(voter_stack.iloc[:,1].max(), candidate_stack.iloc[:,1].max()) + epsilon]

example_xlim = [min(np.min(voter_example[:,0]), np.min(candidate_example[:,0])) - epsilon,
                max(np.max(voter_example[:,0]), np.max(candidate_example[:,0])) + epsilon]
example_ylim = [min(np.min(voter_example[:,1]), np.min(candidate_example[:,1])) - epsilon,
                max(np.max(voter_example[:,1]), np.max(candidate_example[:,1])) + epsilon]


sns.kdeplot(data=voter_stack_sample, x='x', y='y', color = vc, fill=False,
            thresh=0.1, levels=10, alpha = 1, ax = axes[0][0])
sns.kdeplot(data=candidate_stack_sample, x='x', y='y', color = cc, fill=False,
            thresh=0.1, levels=10, alpha = 0.7, ax = axes[0][0])
#axes[0][0].set_title('KDE')
axes[0][0].set_ylabel('')
axes[0][0].set_xlabel('')


axes[0][1].scatter(voter_stack.iloc[:,0], voter_stack.iloc[:,1],
                   facecolors = vc, edgecolors = 'none', alpha = 0.3, s = 10)
axes[0][1].scatter(candidate_stack.iloc[:,0], candidate_stack.iloc[:,1],
                   facecolors = cc, edgecolors = 'none', alpha = 0.01, s = 10)
axes[0][1].set_xlim(scatter_xlim)
axes[0][1].set_ylim(scatter_ylim)
#axes[0][1].set_title('Scatter')


axes[0][2].scatter(voter_example[:,0], voter_example[:,1],
                   facecolors = vc, edgecolors = 'none', alpha = 0.9, s = 30)
axes[0][2].scatter(candidate_example[:,0], candidate_example[:,1],
                   facecolors = cc, edgecolors = 'none', alpha = 0.9, s = 30)
axes[0][2].set_xlim(example_xlim)
axes[0][2].set_ylim(example_ylim)
#axes[0][2].set_title('Example')


for i,E in enumerate(elections_list):
    name = E.__name__
    name_label = name
    if name == 'ChamberlinCourant':
        name_label = 'Chamberlin'
        
    ax_idx = i + 1

    winner_example = result_dict[name][sample_idx]
    winner_stack = pd.DataFrame(np.vstack(result_dict[name]), columns = ['x','y'])
    # Use a smaller sample for KDE plots (otherwise takes a while to run without much added benefit)
    winner_stack_sample = winner_stack.sample(frac=0.25, random_state=42)
    
    sns.kdeplot(data=voter_stack_sample, x='x', y='y', color = vc, fill=False,
                thresh=0.1, levels=10, alpha = 1, ax = axes[ax_idx][0])
    sns.kdeplot(data=winner_stack_sample, x='x', y='y', color = wc, fill=False,
                thresh=0.1, levels=10, alpha = 0.7, ax = axes[ax_idx][0])
    #axes[ax_idx][0].set_ylabel(name_label)
    axes[ax_idx][0].set_ylabel('')
    axes[ax_idx][0].set_xlabel('')
    
    axes[ax_idx][1].scatter(voter_stack.iloc[:,0], voter_stack.iloc[:,1],
                   facecolors = vc, edgecolors = 'none', alpha = 0.3, s = 10)
    axes[ax_idx][1].scatter(winner_stack.iloc[:,0], winner_stack.iloc[:,1],
                       facecolors = wc, edgecolors = 'none', alpha = 0.1, s = 10)
    axes[ax_idx][1].set_xlim(scatter_xlim)
    axes[ax_idx][1].set_ylim(scatter_ylim)

    axes[ax_idx][2].scatter(voter_example[:,0], voter_example[:,1],
                       facecolors = vc, edgecolors = 'none', alpha = 0.9, s = 30)
    axes[ax_idx][2].scatter(winner_example[:,0], winner_example[:,1],
                       facecolors = wc, edgecolors = 'none', alpha = 0.9, s = 30)
    axes[ax_idx][2].set_xlim(example_xlim)
    axes[ax_idx][2].set_ylim(example_ylim)
     

legend_elements = [Line2D([0], [0], marker = 'o', color=vc, lw=2, label='voters'),
                   Line2D([0], [0], marker = 'o', color=cc, lw=2, label='candidates'),
                  Line2D([0], [0], marker = 'o', color=wc, lw=2, label='winners')]

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.075), ncol=3)
plt.savefig(output_file, bbox_inches='tight')