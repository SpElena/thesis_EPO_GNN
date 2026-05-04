# Opinion Dynamics with Graph Neural Networks
This code is in support of the thesis titled
**Opinion Dynamics with Graph Neural Networks: An Application of the EPO Model to Examining Pluralistic Ignorance**

*Completed by Elena Spitcyna*  
Faculty of Applied Mathematics  
Game Theory and Operations Research  
SPbU, 2026  

Supervised by Elena Parilina

This repository contains all code for simulating the binary Expressed-Private Opinion (EPO) model, building and training the GNN-EPO surrogate, and generating the figures and tables presented in the thesis.

## Repository Structure

### Chapter 1 — Binary EPO Model

Implementation and analysis of the original binary EPO model.

| File | Description |
|------|-------------|
| `epo_model.py` | Core EPO model: graph generation, opinion updates, pluralistic ignorance measurement |
| `epo_exp.py` | Batch simulation runner and aggregation functions |
| `epo_plot.py` | Generates PI curves, convergence plots, and LaTeX tables for all parameter sweeps |

To run all Chapter 1 scripts (if you wish to simply confirm the results for example), use:
python run_epo.py

### Chapter 2 — GNN-EPO Surrogate

Graph neural network surrogate of the EPO model with learnable parameter estimation.

| File | Description |
|------|-------------|
| `gnn_epo_model.py` | GNN-EPO architecture: exact surrogate with optional parameter estimators |
| `gnn_epo_train.py` | Training pipeline: data generation, curriculum learning, model training |
| `gnn_epo_eval.py` | Evaluation: exact mode validation, estimator diagnostics, missing-parameter analysis |

To run all Chapter 2 scripts (if you wish to simply confirm the results for example), use:
python run_gnn.py

Note: the training of GNN-EPO generally takes up to 5 minutes depending on your machine. You can bypass the gnn_epo_train.py however by utilizing the trained_EPO_GNN.pth saved in the repository directly (i.e. simply just run chapter_2/gnn_epo_eval.py)
