import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import seaborn as sns


sys.path.append(os.path.join(os.getcwd(), 'metric_voting/code'))
from elections import SNTV,Bloc,STV,Borda, ChamberlinCourant, Monroe, GreedyCC, PluralityVeto, SMRD, OMRD, DMRD, ExpandingApprovals
from tools import group_representation, max_group_representation


# Specify results to plot from:
input_file = 'metric_voting/data/2bloc.npz'

# And where to save them!
output_file1 = 'metric_voting/figures/2bloc_sizes.png'
output_file2 = 'metric_voting/figures/2bloc_sizes_overall.png'


# Read data
loaded_data = np.load(input_file)
result_dict = {key: loaded_data[key] for key in loaded_data.files}


# Specify elections used (and number of samples for each)
elections_list = [SNTV, Bloc, STV, Borda, ChamberlinCourant, GreedyCC, Monroe, PluralityVeto, ExpandingApprovals, SMRD, OMRD, DMRD]
n_samples = 1000


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

####################################################################################################################################
# Compute results

group_sizes = [[100 - i, i] for i in range(0, 105, 5)]
group_select = 1
num_sizes = len(group_sizes)

size_avg_represent = {e.__name__:(np.zeros(num_sizes), np.zeros(num_sizes)) for e in elections_list}
size_avg_represent_overall = {e.__name__:(np.zeros(num_sizes), np.zeros(num_sizes)) for e in elections_list}

for s in range(num_sizes):
    f = 'metric_voting/data/2sizes' + str(s) + '.npz'
    loaded_data = np.load(f)
    result_dict = {key: loaded_data[key] for key in loaded_data.files}
    
    s_avg_represent = {e.__name__:np.zeros(n_samples) for e in elections_list}
    s_avg_represent_overall = {e.__name__:np.zeros(n_samples) for e in elections_list}
    
    for i in range(n_samples):
        voter_positions = result_dict['voters'][i]
        candidate_positions = result_dict['candidates'][i]
        labels = result_dict['labels'][i]
        labels_overall = np.zeros(len(labels))

        for j,E in enumerate(elections_list):
            name = E.__name__
                
            winners = result_dict[name][i]
            represent = group_representation(voter_positions, candidate_positions, labels, winners, group_select, size = None)
            represent_overall = group_representation(voter_positions, candidate_positions, labels_overall, winners, 0, size = None)
            s_avg_represent[name][i] = represent
            s_avg_represent_overall[name][i] = represent_overall
            
    for ename, evals in s_avg_represent.items():
        # BUG fix for now:
        eval_data = evals[evals >= 1]
        size_avg_represent[ename][0][s] = np.mean(eval_data)
        size_avg_represent[ename][1][s] = np.std(eval_data)
        
    for ename, evals in s_avg_represent_overall.items():
        # BUG fix for now:
        eval_data = evals[evals >= 1]
        size_avg_represent_overall[ename][0][s] = np.mean(eval_data)
        size_avg_represent_overall[ename][1][s] = np.std(eval_data)
        
##############################################################################################################
# Plot results

fig,ax = plt.subplots(figsize=(10, 6), dpi = 200)

# Define the area you want to zoom in on (for example, around x = 4 to x = 6)
#x1, x2, y1, y2 = 0.2, 0.3, 0.9, 1.3
# Create the inset of the zoomed area
#axins = inset_axes(ax, width="30%", height="30%", loc='lower right')  # Set the location

Asizes = [x[group_select]/100 for x in group_sizes]
for i, (ename,evals) in enumerate(size_avg_represent.items()):
    if name == 'ChamberlinCourant':
        name_label = 'CC'
    elif name == 'ExpandingApprovals':
        name_label = 'Expanding'
    ax.plot(Asizes[5:], evals[0][5:], label=ename, color = colors[i], linewidth = 3, marker = 'o')
    #ax.fill_between(Asizes, evals[0] - evals[1], evals[0] + evals[1], color=tab20_colors[i], alpha=0.05)
    
    # Plot the same data on the inset axes
    #axins.plot(Asizes, evals[0], label=ename, color = colors[i], linewidth = 1.5, marker = 'o', markersize=2)

'''
# Set the limits for the zoomed-in area
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# Remove x and y axis ticks and labels from inset
axins.tick_params(left=False, right=False, bottom=False, top=False)  # Turn off ticks
axins.set_xticks([])  # Remove x-ticks
axins.set_yticks([])  # Remove y-ticks

# Remove axis labels
axins.set_xlabel('')
axins.set_ylabel('')


# Mark the zoomed area on the main plot
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
'''

#ax.set_ylim(0.9, 3)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
ax.set_ylabel('group-inefficiency')
ax.set_xlabel('Bloc size')
#plt.legend(fontsize = 12, loc = 'upper left')
plt.savefig(output_file1, bbox_inches='tight')
plt.show()


###############################################################################################################

fig,ax = plt.subplots(figsize=(10, 7), dpi = 200)

Asizes = [x[group_select]/100 for x in group_sizes]
for i, (ename,evals) in enumerate(size_avg_represent_overall.items()):
    if name == 'ChamberlinCourant':
        name_label = 'CC'
    elif name == 'ExpandingApprovals':
        name_label = 'Expanding'
    ax.plot(Asizes, evals[0], label=ename, color = colors[i], linewidth = 3, marker = 'o')
    #ax.fill_between(Asizes, evals[0] - evals[1], evals[0] + evals[1], color=tab20_colors[i], alpha=0.05)

plt.ylabel('overall-inefficiency')
plt.xlabel('Bloc size')
#plt.legend(fontsize = 12, loc = 'upper left')

names = [ename for ename in size_avg_represent_overall.keys()]
names = ['CC' if n == 'ChamberlinCourant' else n for n in names]
names = ['Expanding' if n == 'ExpandingApprovals' else n for n in names]
legend_elements = [Line2D([0], [0], marker = 'o', color=colors[i], lw=2, label=names[i]) for i in range(len(names))]

#ax.set_ylim(0.9, 3)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4)
plt.savefig(output_file2, bbox_inches='tight')
plt.show()