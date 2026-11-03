# Reference results

Outputs of the scripts in `../code/`, used by `../code/paper_numbers.py` to recompute every
number of Sec. 5 of the paper.

| Folder | Contents | Script |
|---|---|---|
| `sec51/` | Typed node counts of the benchmark (`node_counts.csv`); cut nodes = holes and Euler identity on 21 919 segmentations (`checks.txt`) | `sec51_node_counts.py`, `sec51_checks.py` |
| `sec52/` | Table 1 (`table1.csv`) | `sec52_rag_classes.py` |
| `sec53/` | Table 2 (`benchmark_perturbations.csv`, `acdc_reference_perturbations.csv`), Table 3 and Fig. 6 (`acdc_predictions_*.csv`), training logs | `sec53_*.py`, `acdc_unet.py`, `acdc_deeplab.py` |
| `sec54/` | Table 4 and times (`e1_query.csv`, `e2_scaling.csv`, `e3_sizes_acdc.csv`), summary (`analysis.txt`) and log-log figure (`fig_size_cost.pdf`) | `sec54_size_cost.py`, `sec54_analyse.py` |
| `synthetic/` | Summary tables of the 1300-image benchmark produced by `../notebooks/02_publication_synthetic_experiments.ipynb` | notebook |
