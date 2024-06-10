# Molecular representations benchmarking
Project for benchmarking the performance of different molecular representations on the task of bioactivity prediction ([Papyrus1K dataset](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00672-x)) and quantum properties predictions ([QM9 dataset](https://moleculenet.org/datasets-1)). [DFT computations](https://purl.stanford.edu/kf921gd3855) are also available for QM9 dataset, so we may explore the usefulness of electron density for developing a novel molecular representations.

## Molecular representations

Classical fingerprints from RDKit:
* Morgan fingerprints
* Topological torsion
* Atom Pairs
* MACCS

Physics-informed reprtesentations from DScribe:
* Coulomb matrix

Geometric Graph Neural networks ([review paper](https://arxiv.org/abs/2312.07511)):
* SchNet
* PamNet

## Algorithms

To make the benchmarking more reliable, different classification algorithms are tested

* Random Forest
* XGBoost
* Fastai Tabular learner

## Environment creation
```bash
conda env create -f env.yml
```
