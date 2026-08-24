# Simulation of a Fully Memristive Reservoir Computing Architecture

[![ETH Zurich](https://img.shields.io/badge/ETH-Zurich-blue)](https://ethz.ch)
[![Bachelor Thesis](https://img.shields.io/badge/Project-Bachelor%20Thesis-informational)](#about-this-project)
[![Python](https://img.shields.io/badge/Python-3-yellow)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)](https://www.tensorflow.org)

A simulation framework for a **fully memristive reservoir computing architecture**. It contains a 4×4 pattern-recognition study using the Final Output Method (FOM) and Continuous Output Method (COM), plus an FOM reservoir study with binarized MNIST samples.

---

## Key Features

- **Memristive crossbar reservoir** — Simulates voltage, current, and conductance evolution
- **FOM and COM** — Compares final and intermediate reservoir readouts
- **Pattern-recognition tasks** — Includes four custom 4×4 patterns and an MNIST-based 28×28 study
- **Readout evaluation** — Trains a non-negative softmax readout and generates evaluation plots

## Project Structure

| Path | Purpose |
|------|---------|
| `Problem_1/` | 4×4 FOM/COM reservoir and readout workflows |
| `Problem_1/raw_data/` | Binary patterns and class labels |
| `Problem_2/` | MNIST-based FOM reservoir simulation |
| `README/` | Technical documentation and user guide |

## Usage

Install the source-derived dependencies and run scripts from the repository root:

```bash
python -m pip install -r requirements.txt

python Problem_1/FOM_reservoir.py
python Problem_1/COM_reservoir.py
python Problem_1/FOM_readout.py
python Problem_1/COM_readout.py
python Problem_2/reservoir.py
```

The readout scripts execute their corresponding reservoir simulation before training. Generated plots and model files are excluded from version control.

## Documentation

See [`README/README_DEFINITIVE.md`](README/README_DEFINITIVE.md) for installation details, file descriptions, configuration, outputs, and reproducibility notes.

## About This Project

This repository accompanies the bachelor thesis **“Simulation of a Fully Memristive Reservoir Computing Architecture”** by Erion Isljami at ETH Zurich.

## Citation

> Erion Isljami, *Simulation of a Fully Memristive Reservoir Computing Architecture*, Bachelor Thesis, ETH Zurich.

## Advisors

- Nadia Jimenez Olalla
- Elias Passerini
- Prof. Dr. Juerg Leuthold

## License

No software license is included. Contact the author regarding reuse.
