import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.append(os.path.join(os.getcwd(), 'metric_voting/code'))
from elections import SNTV,Bloc,STV,Borda, ChamberlinCourant, Monroe, GreedyCC, PluralityVeto, SMRD, OMRD, DMRD,ExpandingApprovals
from tools import group_representation, max_group_representation

# Specify results to plot from:
input_file = 'metric_voting/data/4bloc_5cand.npz'

# And where to save them!
output_file = 'metric_voting/figures/4bloc_5cand.png'


# Read data
loaded_data = np.load(input_file)
result_dict = {key: loaded_data[key] for key in loaded_data.files}


# Specify elections used
elections_list = [SNTV, Bloc, STV, Borda, ChamberlinCourant, GreedyCC, Monroe, PluralityVeto, ExpandingApprovals, SMRD, OMRD, DMRD]


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
    "font.size": 18
})



###########################################################################################################
# Plotting:

fig, axes = plt.subplots(len(elections_list) + 1, 3, figsize=(10, 24), dpi = 200)

for i, ax in enumerate(axes.flat):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')

vc = colors[0]
cc = colors[-2]
wc = colors[5]

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
scatter_xlim = [-5 - epsilon,5 + epsilon]
scatter_ylim = [-5 - epsilon,5 + epsilon]

example_xlim = [-5 - epsilon,5 + epsilon]
example_ylim = [-5 - epsilon,5 + epsilon]


sns.kdeplot(data=voter_stack_sample, x='x', y='y', color = vc, fill=False,
            thresh=0.1, levels=10, alpha = 1, ax = axes[0][0])
sns.kdeplot(data=candidate_stack_sample, x='x', y='y', color = cc, fill=False,
            thresh=0.1, levels=10, alpha = 0.9, ax = axes[0][0])
axes[0][0].set_title('KDE')
axes[0][0].set_ylabel('')
axes[0][0].set_xlabel('')


axes[0][1].scatter(voter_stack_sample.iloc[:,0], voter_stack_sample.iloc[:,1],
                   facecolors = vc, edgecolors = 'none', alpha = 0.3, s = 5)
axes[0][1].scatter(candidate_stack_sample.iloc[:,0], candidate_stack_sample.iloc[:,1],
                   facecolors = cc, edgecolors = 'none', alpha = 0.01, s = 5)
axes[0][1].set_xlim(scatter_xlim)
axes[0][1].set_ylim(scatter_ylim)
axes[0][1].set_title('Scatter')


axes[0][2].scatter(voter_example[:,0], voter_example[:,1],
                   facecolors = vc, edgecolors = 'none', alpha = 0.9, s = 30)
axes[0][2].scatter(candidate_example[:,0], candidate_example[:,1],
                   facecolors = cc, edgecolors = 'none', alpha = 0.9, s = 30)
axes[0][2].set_xlim(example_xlim)
axes[0][2].set_ylim(example_ylim)
axes[0][2].set_title('Example')


for i,E in enumerate(elections_list):
    name = E.__name__
    name_label = name
    if name == 'ChamberlinCourant':
        name_label = 'CC'
    elif name == 'ExpandingApprovals':
        name_label = 'Expanding'
        
    ax_idx = i + 1

    winner_example = result_dict[name][sample_idx]
    winner_stack = pd.DataFrame(np.vstack(result_dict[name]), columns = ['x','y'])
    # Use a smaller sample for KDE plots (otherwise takes a while to run without much added benefit)
    winner_stack_sample = winner_stack.sample(frac=0.25, random_state=42)
    
    
    sns.kdeplot(data=voter_stack_sample, x='x', y='y', color = vc, fill=False,
                thresh=0.1, levels=10, alpha = 1, ax = axes[ax_idx][0])
    sns.kdeplot(data=winner_stack_sample, x='x', y='y', color = wc, fill=False,
                thresh=0.1, levels=10, alpha = 0.7, ax = axes[ax_idx][0])
    axes[ax_idx][0].set_ylabel(name_label)
    axes[ax_idx][0].set_xlabel('')
    
    
    axes[ax_idx][1].scatter(voter_stack_sample.iloc[:,0], voter_stack_sample.iloc[:,1],
                   facecolors = vc, edgecolors = 'none', alpha = 0.3, s = 5)
    axes[ax_idx][1].scatter(winner_stack_sample.iloc[:,0], winner_stack_sample.iloc[:,1],
                       facecolors = wc, edgecolors = 'none', alpha = 0.1, s = 5)
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