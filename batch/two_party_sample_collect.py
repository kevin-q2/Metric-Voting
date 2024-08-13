import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import seaborn as sns
import itertools as it
import pulp
from sklearn.cluster import KMeans
import time

sys.path.append(os.path.join(os.getcwd(), 'metric_voting/code'))
from spatial_generation import Spatial, GroupSpatial
from elections import SNTV,Bloc,STV,Borda, ChamberlinCourant, Monroe, GreedyCC, PluralityVeto, SMRD, OMRD, DMRD
from tools import cost, best_group_cost, worst_group_cost, representativeness, representativeness_ratio, remove_candidates, borda_matrix, group_representation, max_group_representation
from election_sampling import election_sample, samples


# Colors!
pal = sns.color_palette("hls", 8)
tab20_colors = plt.cm.tab20.colors


# Choose number of voters n
# And the number of candidates m
n = 100
m = 20

# And the number of winners for the election
k = 4 

# Means for each of the 2 Gaussian distributions
means = [[0, -1.5], [0, 1.5]]
stds = [0.5, 0.5]  # Standard deviations for each Gaussian
two_party_G = [50,50]  # Group Sizes

voter_params = [{'loc': None, 'scale': None, 'size': 2} for _ in range(len(two_party_G))]
for i,mean in enumerate(means):
    voter_params[i]['loc'] = mean

for i,std in enumerate(stds):
    voter_params[i]['scale'] = std
    
candidate_params = {'low': -3, 'high': 3, 'size': 2}

distance = lambda point1, point2: np.linalg.norm(point1 - point2)

two_party_generator = GroupSpatial(m = m, g = len(two_party_G),
                    voter_dists = [np.random.normal]*len(two_party_G), voter_params = voter_params,
                    candidate_dist = np.random.uniform, candidate_params = candidate_params,
                    distance = distance)


# Generate a profile from random candidate and voter positions
profile, candidate_positions, voter_positions, voter_labels = two_party_generator.generate(two_party_G)


# Define elections
elections_dict = {SNTV:{}, Bloc:{}, STV:{},
                 Borda:{}, ChamberlinCourant:{}, GreedyCC:{}, Monroe:{}, PluralityVeto:{},
                 SMRD:{}, OMRD:{}, DMRD:{'rho': 0.5}}
elections_list = [SNTV, Bloc, STV, Borda, ChamberlinCourant, GreedyCC, Monroe, 
                  PluralityVeto, SMRD, OMRD, DMRD]
n_samples = 10000


# and sample from them
f = 'metric_voting/data/2party_1.npz'


#results_list = samples(n_samples, two_party_generator, elections_dict, [two_party_G], k, dim = 2, filename = f)
#result_dict = results_list[0]

loaded_data = np.load(f)
result_dict = {key: loaded_data[key] for key in loaded_data.files}





#################################################################################################################
# PLOTTING:


fig, axes = plt.subplots(len(elections_list) + 1, 3, figsize=(10, 30), dpi = 200)
plt.rcParams.update({'font.size': 12})

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

sample_idx = 5
voter_example = result_dict['voters'][sample_idx]
voter_stack = pd.DataFrame(np.vstack(result_dict['voters']), columns = ['x','y'])
#voter_stack = voter_stack.sample(frac=0.1, random_state=42)

# Baseline
candidate_example = result_dict['candidates'][sample_idx]
candidate_stack = pd.DataFrame(np.vstack(result_dict['candidates']), columns = ['x','y'])
#candidate_stack = candidate_stack.sample(frac=0.1, random_state=42)

sns.kdeplot(data=voter_stack, x='x', y='y', color = vc, fill=False,
            thresh=0.1, levels=10, alpha = 0.7, ax = axes[0][0])
sns.kdeplot(data=candidate_stack, x='x', y='y', color = cc, fill=False,
            thresh=0.1, levels=10, alpha = 0.7, ax = axes[0][0])

axes[0][0].set_title('KDE')
axes[0][0].set_ylabel('')
axes[0][0].set_xlabel('')

axes[0][1].scatter(voter_stack.iloc[:,0], voter_stack.iloc[:,1],
                   facecolors = vc, edgecolors = 'none', alpha = 0.1, s = 10)
axes[0][1].scatter(candidate_stack.iloc[:,0], candidate_stack.iloc[:,1],
                   facecolors = cc, edgecolors = 'none', alpha = 0.05, s = 10)
axes[0][1].set_title('Scatter')

axes[0][2].scatter(voter_example[:,0], voter_example[:,1],
                   facecolors = vc, edgecolors = 'none', alpha = 0.5, s = 30)
axes[0][2].scatter(candidate_example[:,0], candidate_example[:,1],
                   facecolors = cc, edgecolors = 'none', alpha = 0.9, s = 30)
axes[0][2].set_title('Example')


for i,E in enumerate(elections_list):
    name = E.__name__
    name_label = name
        
    ax_idx = i + 1

    winner_example = result_dict[name][sample_idx]
    winner_stack = pd.DataFrame(np.vstack(result_dict[name]), columns = ['x','y'])
    #winner_stack = winner_stack.sample(frac=0.1, random_state=42)
    
    
    sns.kdeplot(data=voter_stack, x='x', y='y', color = vc, fill=False,
                thresh=0.1, levels=10, alpha = 0.7, ax = axes[ax_idx][0])
    sns.kdeplot(data=winner_stack, x='x', y='y', color = wc, fill=False,
                thresh=0.1, levels=10, alpha = 0.7, ax = axes[ax_idx][0])
    
    
    axes[ax_idx][0].set_ylabel(name_label)
    axes[ax_idx][0].set_xlabel('')
    
    axes[ax_idx][1].scatter(voter_stack.iloc[:,0], voter_stack.iloc[:,1],
                   facecolors = vc, edgecolors = 'none', alpha = 0.05, s = 10)
    axes[ax_idx][1].scatter(winner_stack.iloc[:,0], winner_stack.iloc[:,1],
                       facecolors = wc, edgecolors = 'none', alpha = 0.05, s = 10)

    axes[ax_idx][2].scatter(voter_example[:,0], voter_example[:,1],
                       facecolors = vc, edgecolors = 'none', alpha = 0.5, s = 30)
    axes[ax_idx][2].scatter(winner_example[:,0], winner_example[:,1],
                       facecolors = wc, edgecolors = 'none', alpha = 0.9, s = 30)
     

legend_elements = [Line2D([0], [0], marker = 'o', color=vc, lw=2, label='voters'),
                   Line2D([0], [0], marker = 'o', color=cc, lw=2, label='candidates'),
                  Line2D([0], [0], marker = 'o', color=wc, lw=2, label='winners')]

fig.legend(fontsize = 12, handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.075), ncol=3)


plt.savefig('metric_voting/figures/2party_1.png', bbox_inches='tight')

'''
##############################################################################################################
# USE KNOWN GROUP LABELS:
avg_represent = {e.__name__:np.zeros(n_samples) for e in elections_list}
#avg_represent['DMRD'] = avg_represent.pop('RandomDictator')
#avg_represent['OMRD'] = avg_represent.pop('RandomDictator2')
#avg_represent['CC'] = avg_represent.pop('ChamberlinCourant')

group_select = 0
for i in range(n_samples):
    voter_positions = result_dict['voters'][i]
    candidate_positions = result_dict['candidates'][i]
    labels = result_dict['labels'][i]
    
    for j,E in enumerate(elections_list):
        name = E.__name__
        #if name == 'RandomDictator':
        #    name = 'DMRD'
        #elif name == 'RandomDictator2':
        #    name = 'OMRD'
        #elif name == 'ChamberlinCourant':
        #    name = 'CC'
        winners = result_dict[name][i]
        represent = group_representation(voter_positions, candidate_positions, labels, winners, group_select, size = None)
        avg_represent[name][i] = represent
        
represent_data = [values for values in avg_represent.values()]
represent_labels = [name for name in avg_represent.keys()]

# TREATS EVERYONE AS A SINGLE GROUP:
avg_represent1 = {e.__name__:np.zeros(n_samples) for e in elections_list}
#avg_represent1['DMRD'] = avg_represent1.pop('RandomDictator')
#avg_represent1['OMRD'] = avg_represent1.pop('RandomDictator2')
#avg_represent1['CC'] = avg_represent1.pop('ChamberlinCourant')

group_select = 0
for i in range(n_samples):
    voter_positions = result_dict['voters'][i]
    candidate_positions = result_dict['candidates'][i]
    labels = result_dict['labels'][i]
    labels = np.zeros(len(labels))
    
    for j,E in enumerate(elections_list):
        name = E.__name__
        #if name == 'RandomDictator':
        #    name = 'DMRD'
        #elif name == 'RandomDictator2':
        #    name = 'OMRD'
        #elif name == 'ChamberlinCourant':
        #    name = 'CC'
        winners = result_dict[name][i]
        represent = group_representation(voter_positions, candidate_positions, labels, winners, group_select, size = None)
        avg_represent1[name][i] = represent
        
represent_data1 = [values for values in avg_represent1.values()]
represent_labels1 = [name for name in avg_represent1.keys()]

ylimit = max(max([np.max(r) for r in represent_data]), max([np.max(r) for r in represent_data1]))


######################################################################################################################

plt.rcParams.update({'font.size': 18})
# Create a violin plot
plt.figure(figsize=(16, 6), dpi = 200)
ax = sns.violinplot(data=represent_data, palette = tab20_colors, alpha = 1, width = 0.9)
for violin in ax.collections:
    violin.set_alpha(1)
    
ax.set_ylim(0, ylimit)
plt.xticks(ticks=np.arange(len(elections_list)), labels=represent_labels, rotation=67)
plt.ylabel(r'$\alpha$')
plt.savefig('metric_voting/figures/2party_representation_1.png', bbox_inches='tight')



########################################################################################################################

plt.rcParams.update({'font.size': 18})
# Create a violin plot
plt.figure(figsize=(16, 6), dpi = 200)
ax = sns.violinplot(data=represent_data1, palette = tab20_colors, alpha = 1, width = 0.9)
for violin in ax.collections:
    violin.set_alpha(1)
    
ax.set_ylim(0, ylimit)
plt.xticks(ticks=np.arange(len(elections_list)), labels=represent_labels1, rotation=67)
plt.ylabel(r'$\alpha$')
plt.savefig('metric_voting/figures/2party_representation1_1.png', bbox_inches='tight')



#########################################################################################################################
'''