# MESH - Memory Scaffold with Heteroassociation 

Content-addressable memory (CAM) networks, so-called because stored items can be recalled by partial or corrupted versions of the items, exhibit near-perfect recall of a small number of information-dense patterns below capacity and a 'memory cliff' beyond, such that inserting a single additional pattern results in catastrophic forgetting of all stored patterns. We propose a novel ANN architecture, Memory Scaffold with Heteroassociation (MESH), that gracefully trades-off pattern richness with pattern number to generate a CAM continuum without a memory cliff: Small numbers of patterns are stored with complete information recovery matching standard CAMs, while inserting more patterns still results in partial recall of every pattern, with an information per pattern that scales inversely with the number of patterns. Motivated by the architecture of the Entorhinal-Hippocampal memory circuit in the brain, MESH is a tripartite architecture with pairwise interactions that uses a predetermined set of internally stabilized states together with heteroassociation between the internal states and arbitrary external patterns. We show analytically and experimentally that MESH nearly saturates the total information bound (given by the number of synapses) for CAM networks, invariant of the number of stored patterns, outperforming all existing CAM models.

## Results
The following three folders contain the data for the results presented in the paper: 
- `MESH_results`: results from the complete MESH model e.g., those presented in Figure 4 and Figure 5 in the paper 
- `continuum_results`: results from all the existing models and MESH e.g., those presented in Figure 2 and Figure A.1
- `memscaffold_results`: results from the the memory scaffold part of the MESH network e.g., those presented in Figure 3 

## Source Code
The main directory and the `src` folder contains the souce code for the MESH model. 
- `capacity_assoc.py`: can be used to run the memory scaffold part of the MESH network.
- `capacity_assoc_sensory.py`: can be used to run the whole MESH model.
- `iterativepseudo.py`: verifies that iterative online psudoinverse learning rule provides a good approximation for the MESH model. 
-  `mutualinfo.py`: can be used to reproduce the mutual information (MI) and overlap curves presened in the paper.

## Dependencies
- Python
- Numpy
- Scipy
- Matplotlib

## Terminology
As mentioned in the discussion section of the paper, the feature, label and hidden layers of MESH map naturally to LEC (sensory), MEC (grid cells), and the hippocampus (place cells) respectively. Since the architecture of MESH is Motivated by the architecture of the Entorhinal-Hippocampal memory circuit in the brain, we have used brain-inspired terminology in the source code. Following is the mapping from this terminology to the terminology used in the paper. 

| Source code | Paper |
|-------------|-------|
|N<sub>g</sub>|N<sub>L</sub>|  
|N<sub>p</sub>|N<sub>H</sub>|  
|N<sub>s</sub>|N<sub>F</sub>| 
|gbook        |Label states |
|pbook        |Hidden states |
|sbook        |Feature states |

## Citation
If you find this work useful, we would appreciate if you could cite our paper:

```
@InProceedings{pmlr-v162-sharma22b,
  title = 	 {Content Addressable Memory Without Catastrophic Forgetting by Heteroassociation with a Fixed Scaffold},
  author =       {Sharma, Sugandha and Chandra, Sarthak and Fiete, Ila},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {19658--19682},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/sharma22b/sharma22b.pdf},
  url = 	 {https://proceedings.mlr.press/v162/sharma22b.html},
  abstract = 	 {Content-addressable memory (CAM) networks, so-called because stored items can be recalled by partial or corrupted versions of the items, exhibit near-perfect recall of a small number of information-dense patterns below capacity and a ’memory cliff’ beyond, such that inserting a single additional pattern results in catastrophic loss of all stored patterns. We propose a novel CAM architecture, Memory Scaffold with Heteroassociation (MESH), that factorizes the problems of internal attractor dynamics and association with external content to generate a CAM continuum without a memory cliff: Small numbers of patterns are stored with complete information recovery matching standard CAMs, while inserting more patterns still results in partial recall of every pattern, with a graceful trade-off between pattern number and pattern richness. Motivated by the architecture of the Entorhinal-Hippocampal memory circuit in the brain, MESH is a tripartite architecture with pairwise interactions that uses a predetermined set of internally stabilized states together with heteroassociation between the internal states and arbitrary external patterns. We show analytically and experimentally that for any number of stored patterns, MESH nearly saturates the total information bound (given by the number of synapses) for CAM networks, outperforming all existing CAM models.}
} 
```
