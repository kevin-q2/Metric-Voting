import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from metric_voting.elections import *
from metric_voting.measurements import *

def plot_kde():
    pass 

def plot_scatter():
    pass 


def plot_winner_distribution(results, colors, output_file = None):
    """
    Plots the distribution of winners for each election method.
    
    Args:
        results (dict[str, np.ndarray]): Dictionary with strings as keys and their corresponding
            np array datasets as values. In every result dictionary, 
            the voter data points should be given for the 
            key 'voters' as an array of shape (s, n, dim) where s is the number of random samples,
            n is the number of voters in each sample, and dim is the number of dimensions in the 
            metric space. Likewise, 'candidates' should specify a (s, m, dim) array of candidate 
            data points and 'labels' should specify a (s, m) array of group labels for 
            each candidate sample. 
            
            Then, For each election type in the data, the result dictionary should have a 
            key with the election name and a value as the dataset of winner positions 
            as an array of shape (s, k, dim) where k is the number of winners.
            
        colors (list[str]): Length 3 list of colors to use for voters, candidates, 
            and winners respectively. Colors should be specified in hex format. 
            
        output_file (str, optional): Filepath to save the plot to. If None, the plot will
            be displayed but not saved.
    """
    elections = [_ for _ in results.keys() if _ not in ['voters', 'candidates', 'labels']]
    
    fig, axes = plt.subplots(len(elections) + 1, 3, figsize=(10, 24), dpi = 200)
    
    for i, ax in enumerate(axes.flat):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
    voter_color = colors[0]
    candidate_color = colors[-2]
    winner_color = colors[5]

    
    # Index for the sample used for the example plots:
    sample_idx = 1
    voter_example = result_dict['voters'][sample_idx]
    voter_stack = pd.DataFrame(np.vstack(result_dict['voters']), columns = ['x','y'])
    # Use a smaller sample for KDE plots (otherwise takes a while to run without much added benefit)
    voter_stack_sample = voter_stack.sample(frac=0.001, random_state=42)

    # Just showing the distribution without the election
    candidate_example = result_dict['candidates'][sample_idx]
    candidate_stack = pd.DataFrame(np.vstack(result_dict['candidates']), columns = ['x','y'])
    # Use a smaller sample for KDE plots (otherwise takes a while to run without much added benefit)
    candidate_stack_sample = candidate_stack.sample(frac=0.25, random_state=42)



    # Set x and y limits for scatter and example plots:
    epsilon = 0.5
    ymin = np.min((voter_stack.loc[:,'y'].min(), candidate_stack.loc[:,'y'].min()))
    ymax = np.max((voter_stack.loc[:,'y'].max(), candidate_stack.loc[:,'y'].max()))
    xmin = np.min((voter_stack.loc[:,'x'].min(), candidate_stack.loc[:,'x'].min()))
    xmax = np.max((voter_stack.loc[:,'x'].max(), candidate_stack.loc[:,'x'].max()))

    scatter_xlim = [xmin - epsilon,xmax + epsilon]
    scatter_ylim = [xmin - epsilon,xmax + epsilon]

    example_xlim = [xmin - epsilon,xmax + epsilon]
    example_ylim = [xmin - epsilon,xmax + epsilon]


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
