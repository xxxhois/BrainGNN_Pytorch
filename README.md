# BrainGNN: Graph Neural Network for Brain Network Analysis

A PyTorch implementation of BrainGNN for fMRI-based brain network analysis and classification.

## Overview

This project implements BrainGNN for analyzing brain functional connectivity networks from fMRI data. The example uses the ABIDE (Autism Brain Imaging Data Exchange) dataset for autism spectrum disorder classification.

## Features

- Graph neural network architecture for brain network analysis
- Topological pooling for interpretable feature extraction
- Support for ABIDE dataset (1035 subjects)
- Cross-validation with multiple folds
- TensorBoard visualization
- GPU acceleration support

## Quick Start

### 1. Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate braingnn

# Install PyTorch Geometric extensions
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# Verify installation
python check_compatibility.py
```

### 2. Download and Process Data

```bash
# Download ABIDE dataset (50-200GB, may take hours)
python 01-fetch_data.py

# Process data into graph format
python 02-process_data.py
```

### 3. Train Model

```bash
# Basic training
python 03-main.py

# Custom configuration
python 03-main.py --batchSize 32 --n_epochs 100 --lr 0.001
```

### 4. Monitor Training

```bash
tensorboard --logdir=./log
```

Visit http://localhost:6006 to view training curves.

## Project Structure

```
BrainGNN_Pytorch/
├── 01-fetch_data.py          # Download ABIDE dataset
├── 02-process_data.py         # Process data into graphs
├── 03-main.py                 # Main training script
├── check_compatibility.py     # Verify dependencies
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment
├── INSTALL.md                 # Installation guide
├── TRAINING_TIPS.md           # Training recommendations
├── UPDATES.md                 # Changelog
├── data/
│   └── subject_ID.txt         # Subject IDs
├── imports/
│   ├── ABIDEDataset.py        # Dataset loader
│   ├── preprocess_data.py     # Data preprocessing
│   ├── read_abide_stats_parall.py  # Parallel data reading
│   ├── utils.py               # Utility functions
│   └── gdc.py                 # Graph diffusion convolution
├── net/
│   ├── braingnn.py            # Main network architecture
│   ├── braingraphconv.py      # Graph convolution layer
│   └── brainmsgpassing.py     # Message passing layer
├── model/                     # Saved models (gitignored)
└── log/                       # Training logs (gitignored)
```

## Training Configuration

### Basic Parameters

```bash
python 03-main.py \
  --batchSize 32 \
  --n_epochs 100 \
  --lr 0.001 \
  --weightdecay 0.005
```

### Regularization Parameters

```bash
--lamb0 1      # Classification loss weight
--lamb3 0.1    # Entropy regularization
--lamb4 0.1    # Entropy regularization
--lamb5 0.1    # Consistency regularization
```

See [TRAINING_TIPS.md](TRAINING_TIPS.md) for detailed hyperparameter tuning guide.

## Expected Performance

On ABIDE dataset:
- Training Accuracy: 70-80%
- Test Accuracy: 65-75%
- Training Time: 2-4 hours (GPU)

## Requirements

- Python 3.8+
- PyTorch 1.13+
- PyTorch Geometric
- CUDA 11.7 (for GPU)
- 16GB+ RAM
- 50GB+ disk space (for full dataset)

## Citation

If you use this code, please cite:

```bibtex
@article{li2020braingnn,
  title={Braingnn: Interpretable brain graph neural network for fmri analysis},
  author={Li, Xiaoxiao and Zhou, Yuan and Dvornek, Nicha and Zhang, Muhan and Gao, Siyuan and Zhuang, Juntang and Scheinost, Dustin and Staib, Lawrence and Ventola, Pamela and Duncan, James},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License

This project is licensed under the GNU General Public License v3.0.

## Acknowledgments

- Original BrainGNN implementation
- ABIDE dataset contributors
- PyTorch Geometric team

## Troubleshooting

### Common Issues

1. **CUDA not available**: Install CUDA-enabled PyTorch
2. **Memory errors**: Reduce batch size or use data subset
3. **Import errors**: Check `check_compatibility.py`

See [INSTALL.md](INSTALL.md) for detailed troubleshooting.

## Contact

For questions and issues, please open a GitHub issue.
