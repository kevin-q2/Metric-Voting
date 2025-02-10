import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import random
import seaborn as sns
from metric_voting import *


colors = ["#0099cd","#ffca5d","#00cd99","#99cd00","#cd0099","#9900cd","#8dd3c7",
        "#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5","#bc80bd",
        "#ccebc5","#ffed6f","#ffffb3","#a6cee3","#1f78b4","#b2df8a","#33a02c",
        "#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a","#b15928",
        "#64ffda","#00B8D4","#A1887F","#76FF03","#DCE775","#B388FF","#FF80AB",
        "#D81B60","#26A69A","#FFEA00","#6200EA",
    ]

#colors = colors[:6] + colors[-12::2]

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.serif": [],
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 22
})

input_file = 'data/four_bloc/four_winner/samples.npz'
output_file = 'figures/four_bloc/four_winner/distribution_tiebreak.png'

loaded_data = np.load(input_file)
result_dict = {key: loaded_data[key] for key in loaded_data.files}
n_samples = result_dict['voters'].shape[0]


# Chamberlin tiebreak samples:
f = 'data/four_bloc/four_winner/tiebreak/samples_chamberlin'
chamberlin_result_dict = None

for i in range(20):
    fname = f + str(i) + '.npz'
    loaded_data = np.load(fname)
    result_dict_i = {key: loaded_data[key] for key in loaded_data.files}
    if chamberlin_result_dict is None:
        chamberlin_result_dict = result_dict_i
    else:
        for election, winner_array in result_dict_i.items():
            chamberlin_result_dict[election] = np.vstack((
                chamberlin_result_dict[election],
                winner_array
            ))


# Monroe Tiebreak samples:
f = 'data/four_bloc/four_winner/tiebreak/samples_monroe'
monroe_result_dict = None

for i in range(20):
    fname = f + str(i) + '.npz'
    loaded_data = np.load(fname)
    result_dict_i = {key: loaded_data[key] for key in loaded_data.files}
    if monroe_result_dict is None:
        monroe_result_dict = result_dict_i
    else:
        for election, winner_array in result_dict_i.items():
            monroe_result_dict[election] = np.vstack((
                monroe_result_dict[election],
                winner_array
            ))
            
            
result_dict['ChamberlinCourant'] = chamberlin_result_dict['ChamberlinCourantTiebreak']
result_dict['Monroe'] = monroe_result_dict['MonroeTiebreak']


plot_winner_distribution(
    results = result_dict,
    fig_params = {'figsize' : (10, 24), 'dpi' : 200},
    colors = [colors[0], colors[10], colors[4]],
    xlim = [-4,4],
    ylim = [-4,4],
    sample_fraction = 0.25,
    kde_sample_fraction = 0.5,
    random_seed = 42,
    output_file = output_file
)