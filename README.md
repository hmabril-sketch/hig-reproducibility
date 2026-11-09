# HIG reproducibility

Code, trained weights and reference results for

> F. Díaz-del-Río, M. J. Morón-Fernández, H. Molina-Abril, P. Real.
> *Homological Incidence Graph (HIG): a topology-aware representation for segmented images.*
> Submitted to the Journal of Mathematical Imaging and Vision.

The repository reproduces every experiment of Section 5 of the paper: the implementation checks
(5.1), the comparison with region adjacency graphs (5.2, Table 1), the topological comparison of
segmentations on the synthetic benchmark and on cardiac MRI (5.3, Tables 2 and 3, Fig. 6), and
the size and cost measurements (5.4, Table 4).

| Folder | Contents |
|---|---|
| `code/` | HIG construction (reference and compiled), comparisons, and one script per experiment |
| `results/` | Reference output of every experiment (CSV) |
| `weights/` | Trained U-Net (epochs 2, 5, 16) and DeepLabv3+ (epochs 5, 10, 16) |
| `figures/` | Fig. 6 |
| `notebooks/` | Interactive notebooks: 10x10 HIG demo, rectangle generator, synthetic benchmark |
| `earlier_experiments/` | Experiments of an earlier version of the work (BreaKHis, generic images); not part of the paper |

## Quick check (a few minutes, no data needed)

```bash
pip install -r requirements.txt        # or: numpy scipy pandas networkx numba Pillow for this check
cd code
python smoke_test.py                   # HIG of Fig. 1 = matrices of Fig. 4; compiled = reference construction
python paper_numbers.py                # recomputes 94 numbers of Sec. 5 from results/ and compares them with the paper
python sec52_rag_classes.py /tmp/table1.csv     # Table 1 from scratch (about 15 s)
```

`paper_numbers.py` reads only the CSV files in `results/`. Point it to another directory
(`python paper_numbers.py OTHER_RESULTS_DIR`) to check results regenerated with the scripts below.
The same checks run on every push (`.github/workflows/smoke-test.yml`).

## From the paper to the code

All scripts are in `code/` and are run from that folder. Times are for two CPU cores.

| Paper | Script | Reference output | Time |
|---|---|---|---|
| Sec. 5.1, node counts of the benchmark | `sec51_node_counts.py OUT.csv` | `results/sec51/node_counts.csv` | 10 s |
| Sec. 5.1, cut nodes = (4,8) holes and Euler identity on the 21 919 segmentations | `sec51_checks.py [RUN_DIR ...]` | `results/sec51/checks.txt` | 1 min (benchmark), 2 min (all) |
| Sec. 5.2, Table 1 | `sec52_rag_classes.py OUT.csv` | `results/sec52/table1.csv` | 15 s |
| Sec. 5.3, Table 2 (benchmark rows) | `sec53_benchmark_perturbations.py OUT.csv` | `results/sec53/benchmark_perturbations.csv` | 3 min |
| Sec. 5.3, Table 2 (ACDC rows) | `sec53_acdc_reference_perturbations.py RUN_DIR OUT.csv` | `results/sec53/acdc_reference_perturbations.csv` | 10 min |
| Sec. 5.3, networks | `acdc_unet.py`, `acdc_deeplab.py` (training) or `acdc_predict.py` (from `weights/`) | `results/sec53/*_train.log` | 50 min / 1 h / 5 min |
| Sec. 5.3, cleaned predictions | `clean_predictions.py RUN_DIR OUT_DIR` | | 1 min |
| Sec. 5.3, Table 3 and text | `sec53_acdc_predictions.py RUN_DIR TAG NET CLEAN OUT.csv` | `results/sec53/acdc_predictions_*.csv` | 2.5 min per epoch |
| Sec. 5.3, Fig. 6 | `sec53_fig6.py UNET_RUN UNET_CLEAN_RUN OUT.png` | `figures/fig6_acdc_qc.png` | 5 s |
| Sec. 5.4, compiled = reference construction (2045 segmentations) | `sec54_verify_fast.py [UNET_RUN]` | | 1.5 min |
| Sec. 5.4, Table 4, times | `sec54_size_cost.py UNET_RUN OUT_DIR`, then `sec54_analyse.py OUT_DIR` | `results/sec54/` | < 1 h |
| Sec. 5, all numbers | `paper_numbers.py [RESULTS_DIR]` | | 10 s |
| Regenerated against reference results | `compare_results.py NEW_DIR` | | 10 s |

Times in `results/sec54/` were measured on one core of an Intel Xeon at 2.1 GHz; they depend on
the machine, so `paper_numbers.py` will report them as different if Sec. 5.4 is re-run elsewhere.
Every other output is deterministic: the scripts fix all random seeds.

### Library modules (`code/`)

| Module | Contents |
|---|---|
| `hig_construct.py` | Reference implementation of Algorithm 1: regions, frontier graph, geometric 1-nodes, cut nodes and carriers, junctions and virtual 0-nodes, `HIGM_{0,1}` and `HIGM_{1,2}` |
| `hig_fast.py` | Compiled (numba) implementation of Algorithm 1, identical output |
| `topoqc.py` | Typed HIG as a graph, per-class Betti numbers, RAG with `OUT`, Dice, isomorphism tests and localized differences |
| `rag8.py` | RAG with corner contacts and `OUT` |
| `bettimatch.py` | Per-class Betti matching error |
| `euler_regions.py` | Holes of every region from its Euler number (Gray's bit quads), independent of the HIG |
| `queries.py` | Holes of every region from the HIG, from the RAG with corner contacts (low-link search or region by region), and by flood fill |
| `rectgen.py`, `benchmark.py` | Structured rectangle generator; the 1298 benchmark images and the five perturbations |
| `acdc_perturbations.py` | The five perturbations on the ACDC references |
| `acdc_unet.py`, `acdc_deeplab.py`, `acdc_predict.py` | Networks, training and prediction |

Conventions, as in the paper: regions are 4-connected same-label components; complements and
holes are 8-connected; `OUT` is the image exterior; region nodes are numbered in raster order of
their first pixel and `OUT` comes last.

## Installation

Python 3.10 or later. The results were produced with Python 3.13, NumPy 2.5, SciPy 1.18,
pandas 3.0, NetworkX 3.6, numba 0.68, scikit-image 0.26, PyTorch 2.14 (CPU) and nibabel 5.4.

```bash
pip install -r requirements.txt
```

**Betti matching** (Tables 2 and 3) uses the library of Stucki et al.,
[Betti-Matching-3D](https://github.com/nstucki/Betti-Matching-3D) (we used commit `db623be`):

```bash
git clone https://github.com/nstucki/Betti-Matching-3D.git
cd Betti-Matching-3D && git checkout db623be
mkdir build && cd build
cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && make
export BETTI_MATCHING_BUILD=$PWD          # folder that contains betti_matching*.so
```

Every script that computes Betti matching accepts `--no-bm` to skip it.

## ACDC data

The cardiac experiments use the ACDC data set (Bernard et al., IEEE TMI 2018), which is not
redistributed here; it is available from the challenge organisers
(https://www.creatis.insa-lyon.fr/Challenge/acdc/). We used a preprocessed version in which the
end-diastolic and end-systolic frames are resampled to 1 x 1 mm in plane, cropped to 192 x 192
pixels and stored as 8-bit NIfTI files:

```text
DATA_DIR/<any subfolders>/patientXXX_sax_ed.nii.gz      patientXXX_sax_ed_gt.nii.gz
                          patientXXX_sax_es.nii.gz      patientXXX_sax_es_gt.nii.gz
```

<!-- TODO (authors): say where this preprocessed version comes from, or how it was produced. -->

Patients 1-100 are used for training and 101-150 for testing (916 test slices, 854 of which
contain a cardiac structure). Labels: 0 background, 1 RV, 2 MYO, 3 LV.

### Reproducing the ACDC experiments

```bash
cd code
export BETTI_MATCHING_BUILD=/path/to/Betti-Matching-3D/build
# predictions from the trained weights (bit-identical to those of the paper), or retrain with
# acdc_unet.py DATA_DIR runs/unet and acdc_deeplab.py DATA_DIR runs/deeplab
python acdc_predict.py unet    DATA_DIR runs/unet    ../weights/unet_ep02.pt ../weights/unet_ep05.pt ../weights/unet_ep16.pt
python acdc_predict.py deeplab DATA_DIR runs/deeplab ../weights/deeplab_ep05.pt ../weights/deeplab_ep10.pt ../weights/deeplab_ep16.pt
python clean_predictions.py runs/unet    runs/unet_clean
python clean_predictions.py runs/deeplab runs/deeplab_clean

mkdir -p out/sec51 out/sec52 out/sec53 out/sec54
python sec51_node_counts.py out/sec51/node_counts.csv
python sec51_checks.py runs/unet runs/unet_clean runs/deeplab runs/deeplab_clean > out/sec51/checks.txt
python sec52_rag_classes.py out/sec52/table1.csv
python sec53_benchmark_perturbations.py out/sec53/benchmark_perturbations.csv
python sec53_acdc_reference_perturbations.py runs/unet out/sec53/acdc_reference_perturbations.csv
python sec53_acdc_predictions.py runs/unet          unet          U-Net      0 out/sec53/acdc_predictions_unet.csv
python sec53_acdc_predictions.py runs/unet_clean    unet_clean    U-Net      1 out/sec53/acdc_predictions_unet_clean.csv
python sec53_acdc_predictions.py runs/deeplab       deeplab       DeepLabv3+ 0 out/sec53/acdc_predictions_deeplab.csv
python sec53_acdc_predictions.py runs/deeplab_clean deeplab_clean DeepLabv3+ 1 out/sec53/acdc_predictions_deeplab_clean.csv
python sec53_fig6.py runs/unet runs/unet_clean out/fig6.png
python sec54_verify_fast.py runs/unet
python sec54_size_cost.py runs/unet out/sec54 && python sec54_analyse.py out/sec54
python compare_results.py out          # row by row against results/ (times excluded)
python paper_numbers.py out
```

## Output columns

Every row of `results/sec53/*.csv` is one pair (reference, test segmentation). `dice` is the
mean Dice coefficient over the classes of the reference, background included; `betti_changed`,
`rag_changed` (RAG with `OUT`), `rag8_changed` (RAG with corner contacts and `OUT`) and
`hig_changed` (typed HIG, preserving labels, node types and multiplicities) say whether the two
segmentations differ; `bm0`, `bm1` and `bm` are the per-class Betti matching errors in dimensions
0, 1 and in total; `n_diffs` is the number of localized differences, `n_diffs_near_change` the
number within one pixel of a changed pixel, and `diff_region`, `diff_contact`, `diff_hole`,
`diff_junction` count them by node type. In `acdc_predictions_*.csv`, `gt_*` and `pred_*`
describe the anatomical constraints of the reference and of the prediction (`*_plausible`).

## Notebooks

| Notebook | Colab |
|---|---|
| Interactive 10x10 HIG | [Open](https://colab.research.google.com/github/hmabril-sketch/hig-reproducibility/blob/main/notebooks/00_HIG_10x10_demo.ipynb) |
| Structured rectangle generator | [Open](https://colab.research.google.com/github/hmabril-sketch/hig-reproducibility/blob/main/notebooks/01_rectangle_synthetic_generator.ipynb) |
| Synthetic benchmark, node counts (`results/synthetic/`) | [Open](https://colab.research.google.com/github/hmabril-sketch/hig-reproducibility/blob/main/notebooks/02_publication_synthetic_experiments.ipynb) |
| HIG editors, 20x20 | `notebooks/HIG4_8_ed_20x20_Colab_v2.ipynb`, `notebooks/HIGM48_editor_20x20_VISUAL.ipynb` |

## Earlier experiments

`earlier_experiments/` keeps the BreaKHis 40X and generic real-image pipelines of an earlier
version of this work, with their own notebooks, scripts and results. They are not used in the
paper.
