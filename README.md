# Computer Vision Expirements
Package name: `cv_expt`.

## Directory Structure
```
|── data
│   └── # Directory for storing raw and processed datasets.
├── setup.py
│   └── # Setup script for installing dependencies and setting up the project.
├── cv_expt
│   ├── base
│   │   ├── defs
│   │   │   └── configs.py
│   │   │       # Configuration definitions for experiments.
│   │   ├── data
│   │   │   └── base_dataset.py
│   │   │       # Base class for dataset handling and preprocessing.
│   │   ├── trainer
│   │   │   └── base_trainer.py
│   │   │       # Base class for training routines and loops.
│   │   ├── logger
│   │   │   └── base_logger.py
│   │   │       # Base class for logging experiment results - and metrics.
│   │   └── models
│   │       └── base_model.py
│   │           ├── Implements training and inference modes.
│   │           └── Contains input/output processing logic for models.
│   ├── vis
│   │   └── visualization.py
│   │       # Visualization utilities for experiment results.
│   ├── logger
│   │   ├── local_logger.py
│   │   │   # Logger for saving logs locally.
│   │   └── wandb_logger.py
│   │       # Logger for integrating with Weights & Biases (WandB).
│   └── experiment
│       └── experiment.py
│           # Script to define and run experiments.
├── notebooks
│   ├── expt1_name.ipynb
│   │   # Jupyter notebook for Experiment 1.
│   └── expt2_name.ipynb
│       # Jupyter notebook for Experiment 2.
├── assets
│   └── # Directory for storing images and files used in the README or documentation.
├── models
│   └── # Directory to store model weight files for logging and evaluation.
├── outputs
│   ├── # Directory for storing local outputs, not committed to the repository.
│   └── results
│       ├── expt1
│       │   ├── logs
│       │   │   └── run_logs.logs
│       │   │       # Log files for tracking the progress and results of Experiment 1.
│       │   ├── epoch_0.png
│       │   │   # Sample output image from the first epoch of Experiment 1.
│       │   └── best_model.pth
│       │       # The best-performing model checkpoint from Experiment 1.

```



## Experiments
- [x] [Can U-Nets perform Grayscale to RGB?](/notebooks/grayscal_to_rgb.ipynb)
- [ ] Can U-Nets perform histogram equalization?
- [ ] Can U-Nets perform image completion?
