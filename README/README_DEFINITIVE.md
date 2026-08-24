# Technical Documentation & User Guide

## About This Project

This repository accompanies the bachelor thesis **“Simulation of a Fully Memristive Reservoir Computing Architecture”** by Erion Isljami at ETH Zurich.

**Supervisors:** Elias Passerini, Nadia Jimenez Olalla, Prof. Dr. Juerg Leuthold

## Project Overview

The code simulates a reservoir computing system in which a memristive crossbar acts as the dynamical reservoir. Binary image columns are converted into voltage pulses, the pulses update crossbar conductances, and column currents are converted into output voltages.

The repository contains two studies:

1. **Problem 1:** four-class recognition of 4×4 binary patterns using Final Output Method (FOM) and Continuous Output Method (COM), followed by a trainable readout.
2. **Problem 2:** an FOM reservoir simulation using binarized 28×28 MNIST samples of digits 0–3.

## Repository Contents

| Path | Purpose |
|------|---------|
| `Problem_1/FOM_reservoir.py` | 4×4 FOM crossbar simulation |
| `Problem_1/COM_reservoir.py` | 4×4 COM crossbar simulation |
| `Problem_1/FOM_readout.py` | FOM readout training and evaluation |
| `Problem_1/COM_readout.py` | COM readout training and evaluation |
| `Problem_1/functions.py` | Graph, pulse, conductance, dataset, and plotting helpers |
| `Problem_1/raw_data/` | Four binary patterns, class labels, and reference inverses |
| `Problem_2/reservoir.py` | MNIST-based FOM reservoir simulation |
| `Problem_2/functions.py` | Reservoir and plotting helpers for Problem 2 |

## Installation

The original code did not record package versions. `requirements.txt` therefore lists the imported dependencies without version pins.

```bash
git clone https://github.com/erionloni/Simulation-of-a-Fully-Memristive-Reservoir-Computing-Architecture.git
cd Simulation-of-a-Fully-Memristive-Reservoir-Computing-Architecture

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies: NumPy, NetworkX, Matplotlib, Seaborn, TensorFlow/Keras, and scikit-learn.

## Usage

Run all commands from the repository root because the original scripts use repository-relative paths.

```bash
# Reservoir simulations
python Problem_1/FOM_reservoir.py
python Problem_1/COM_reservoir.py

# Complete reservoir and readout workflows
python Problem_1/FOM_readout.py
python Problem_1/COM_readout.py

# MNIST-based reservoir simulation
python Problem_2/reservoir.py
```

The readout modules import and execute the corresponding reservoir module before training.

## Methods

### Final Output Method

FOM reads the crossbar after the complete input pattern has been applied. The final column currents are converted to output voltages.

### Continuous Output Method

COM reads the crossbar during pattern application. Intermediate output voltages are combined into the readout feature vector.

### Readout Layer

Problem 1 uses a non-negative dense softmax layer with categorical cross-entropy and Adam. The scripts compare standardized and non-standardized reservoir outputs.

### MNIST Study

Problem 2 loads MNIST through Keras, selects the first training sample for digits 0–3, and binarizes each image at a threshold of 128.

## Data and Outputs

`pattern.txt` contains four 4×4 patterns: main diagonal, anti-diagonal, horizontal line, and vertical line. `pattern_class.txt` contains labels `0 1 2 3`. `pattern_inv.txt` records the complemented patterns; the scripts also compute these complements at runtime.

The simulations generate:

- Conductance-evolution and crossbar-state plots
- Conductance matrices and output-voltage plots
- Accuracy, loss, weight, and confusion-matrix plots for Problem 1
- Keras readout models for the first standardized and non-standardized runs

Generated outputs and models are excluded by `.gitignore`.

## Included Configuration

| Setting | Value |
|---------|-------|
| Crossbar size | 4×4 in Problem 1; 28×28 in Problem 2 |
| Read voltage | 0.1 V |
| Pulse amplitude | 5 V |
| Read resistance | 100 Ω |
| Readout training/test datasets | 100 / 20 per run |
| Readout epochs and batch size | 100 / 10 |
| Repeated readout runs | 2 per case |
| Bias voltage | 0.5 V |

Memristor-model constants and timing parameters are defined at the top of each reservoir script.

## Reproducibility Notes

- No explicit NumPy or TensorFlow random seed is set, so readout results may vary.
- Problem 2 may download MNIST through Keras on first use.
- No completed numerical result set was included in the supplied source snapshot; this repository therefore makes no additional performance claims.
- The published Python and text data files are byte-identical copies of the supplied originals.

Local `.DS_Store` files and Python bytecode caches were excluded. No credentials, personal paths, private datasets, or large generated files were included.

## Citation and License

> Erion Isljami, *Simulation of a Fully Memristive Reservoir Computing Architecture*, Bachelor Thesis, ETH Zurich.

No software license is included. Contact the author regarding reuse.
