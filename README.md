# Metric Voting

A computational implementation of multi-winner social choice mechanisms in a metric setting. 
We assume that voters and candidates are embedded in some latent metric space and provide
code for randomly generating voter and candidate positions given some input distributions. 
We also implement many different election mechanisms and
provide tools for measuring voter representation in their output. 

## Getting Started


## Social Choice Mechanisms Supported:
We provide implementations of the following multi-winner election mechansisms, all 
of which can be found in `elections.py`. For 
more information and intellectual stimulation please refer to [1, 2]

1. SNTV (Plurality)
2. Bloc
3. STV
4. Borda
5. Chamberlin Courant
6. A greedy Chamberlin Courant Approximation
7. Monroe
8. Plurality Veto [3]
9. Expanding Approvals [4, 5]
10. Variations on multi-winner Random Dictator

## Examples
All examples for generating metric settings, performing elections, sampling, and visualizing results 
can be found in `examples.ipynb`. Reading through this notebook will hopefully help 
with learning how to interface with the spatial generation tools in `spatial_generation.py`, 
the mechanisms in `elections.py`, the sampling procedure used in `election_sampling.py`, 
and the measures of representation in `tools.py`.


## Experimentation
We include collected samples from a few of our experiments in `data/`. All code for data collection,
including for more intensive experiments which we omit from the data folder, can be found in the 
`batch/` folder. Along with it, are a messy set of files that we used for plotting results for our 
experiments (The KDE plots can take a while to run!). This is messy for now...I plan to clean things up. 

## References
[1] Faliszewski, Piotr, et al. "Multiwinner voting: A new challenge for social choice theory." Trends in computational social choice 74.2017 (2017): 27-47.

[2] Elkind, Edith, et al. "What do multiwinner voting rules do? An experiment over the two-dimensional euclidean domain." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 31. No. 1. 2017.

[3] Kizilkaya, Fatih Erdem, and David Kempe. "Plurality veto: A simple voting rule achieving optimal metric distortion." arXiv preprint arXiv:2206.07098 (2022).

[4] Kalayci, Yusuf, David Kempe, and Vikram Kher. "Proportional representation in metric spaces and low-distortion committee selection." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 9. 2024.

[5] Aziz, Haris, and Barton E. Lee. "The expanding approvals rule: improving proportional representation and monotonicity." Social Choice and Welfare 54.1 (2020): 1-45.
