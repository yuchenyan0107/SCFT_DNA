# SCFT_DNA

In this project we modeled the chromosome organization with self-consistent field theory of polymers. 

A work-in-progress version of the full report can be found in `docs/MEP_report_draft_11Dec.pdf`, where more details about the algorithms and background are introduced.

## Project Structure

The code in this repository contains a full workflow: 
inferring the chromatin "binding sites" from Hi-C map; 
calculating polymer field with self-consistent field theory (SCFT);
and result interpretation of polymer conformation.

- `Bead_sequence_MC/`- Binding sites inferring
  - `chromatin_binding_sites_infer.ipynb` - The Jupyter Notebook that shows the workflow
  - `data/` - Contain the Hi-C matrix, theoretical contact profile, and h3k27 & rnaseq data (not used for now)
  - `mf_chromatin_functions/` - Functions used in the Jupyter Notebook
- `SCFT_DNA/` - Self-consistent field theory and result analysis
  - `SCFT_functions/`- Functions related to SCFT computation
    - `computation_functions.py` - Computing field, density, mixing, and free energy
    - `QSR_solvers.py` - MDE pseudo-spectrum solvers; periodical BC for 2D and 3D, and no-flux BC for 3D
    - `QSR_dct.py` - no-flux BC for 2D
    - `loop_functions.py` - SCFT iteration process
  - `DNA_input/` import sequence inferred to SCFT 
  - Jupyter notebooks that contains SCFT in different demo cases

## Project introduction

In this project, we implemented a two-part computational pipeline that connects the Hi-C maps to chromosomal spatial distributions during the interphase. 
The first part uses Monte Carlo method to infer the chromatin “binding-site” sequence (figure 1) which represents the effective interaction pattern along the locus that gives rise to the Hi-C map. 
The second part uses self-consistent field theory (SCFT) of polymers to predict the spatial distribution of a chromatin polymer (2M bp), with the sequence inferred in the first part.

![workflow_MC](docs/figures/workflow_MC.jpg)

*Figure 1: Workflow of the Monte Carlo process that infers the binding-site sequence. 
The predict contact matrix constructed from the inferred sequence the resembles closely the Hi-C map.*

The SCFT translates pairwise interactions between polymer segments into a chemical potential field and its conjugated polymer probability-density distribution (Figure 2a). 
In our model, the chromatin (2M bp) is represented as a Gaussian chain, and the field is calculated from the Flory-Huggins mixing of the polymer segment types obtained from the part I. 
From the SCFT solution, we computed the polymer density and sampled a representative single-chain conformation by following the positions of maximum polymer contour density (Figure 2c, d). 
The contact map (contour density overlap map) of this conformation shows a high similarity to the experimental median-distance map (Figure 2b)

![result_combined](docs/figures/SCFT_results_combined.jpg)

*Figure 2: a) Schematic representation of SCFT framework. 
b) Comparison of median-distance map (left) and the contact map calculate from the SCFT result (right). 
c) Polymer density and most-probable single chain conformation (dark line). 
d) One-dimensional slices of the contour probability density. The positions of the two slices are indicated in (c).*

## 

## Disclaimer: AI usage

Due to the time constrain, some of the code are implemented by ChatGPT or other AI models. 
This includes the data import & preprocessing in the sequence inferring part, and some of the plots for the SCFT result analysis.

However, the core features, including the SCFT algorithm and the simulated annealing Monte Carlo process, were implemented without any AI model.

## Other

The Monte Carlo method for finding the binding site sequence is largely inspired by

https://github.com/marianoimperatore/MeanFieldChromatin

and

Bianco, S., Lupiáñez, D. G., Chiariello, A. M., Annunziatella, C., Kraft, K., Schöpflin, R., Wittler, L., Andrey, G., Vingron, M., Pombo, A., Mundlos, S., & Nicodemi, M. (2018). Polymer physics predicts the e]ects of structural variants on chromatin architecture. Nature Genetics, 50(5), 662–667. https://doi.org/10.1038/s41588-018-0098-8
