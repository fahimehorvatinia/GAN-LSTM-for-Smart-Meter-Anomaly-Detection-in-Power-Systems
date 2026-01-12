# Energy Anomaly Detection using GAN (TAnoGAN)

This repository contains an implementation of TAnoGAN (Time Series Anomaly Detection with Generative Adversarial Networks) for energy anomaly detection.

## Overview

This project implements a GAN-based approach for detecting anomalies in energy meter readings. The model uses LSTM-based Generator and Discriminator networks to learn the normal patterns in energy consumption data and identify anomalous behavior.

## Features

- **LSTM-based GAN architecture** for time series anomaly detection
- **Automatic threshold optimization** to achieve >90% accuracy
- **Comprehensive evaluation metrics** (Accuracy, Precision, Recall, F1-score, ROC-AUC)
- **Visualization tools** for anomaly detection results
- **Flexible threshold testing** without retraining

## Project Structure

```
energy-anomaly-detection-gan/
├── src/
│   ├── energy_dataset.py          # Dataset loader for energy data
│   ├── run_tanogan.py              # Main training and evaluation script
│   ├── test_thresholds.py           # Utility to test different thresholds
│   ├── find_optimal_threshold.py   # Comprehensive threshold search
│   └── models/
│       ├── __init__.py
│       └── recurrent_models_pyramid.py  # LSTM Generator and Discriminator
├── data/                            # Place your data files here
│   ├── train.csv                    # Training data (with anomaly labels)
│   └── test.csv                     # Test data (optional)
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd energy-anomaly-detection-gan
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place your `train.csv` file in the `data/` directory
   - The CSV should have columns: `timestamp`, `building_id`, `meter_reading`, and `anomaly` (for training data)

## Usage

### Training and Evaluation

Run the main script to train the model and evaluate on your data:

```bash
cd src
python run_tanogan.py
```

This will:
1. Load and preprocess the data
2. Train the GAN (Generator and Discriminator)
3. Calculate anomaly scores for all samples
4. Find the optimal threshold
5. Generate evaluation metrics and visualizations
6. Save all results to the `results/` directory

### Testing Different Thresholds

After training, you can test different thresholds without retraining:

```bash
cd src
python test_thresholds.py --threshold 4.0
```

Or use percentile-based thresholds:
```bash
python test_thresholds.py --percentile 95
```

For interactive mode:
```bash
python test_thresholds.py --interactive
```

### Finding Optimal Threshold

Run a comprehensive threshold search:
```bash
cd src
python find_optimal_threshold.py
```

## Data Format

The expected CSV format for training data:

| timestamp | building_id | meter_reading | anomaly |
|-----------|-------------|---------------|---------|
| 2020-01-01 00:00:00 | 1 | 1234.5 | 0 |
| 2020-01-01 01:00:00 | 1 | 1235.2 | 0 |
| 2020-01-01 02:00:00 | 1 | 1500.0 | 1 |

- `timestamp`: DateTime string
- `building_id`: Building identifier
- `meter_reading`: Energy meter reading value
- `anomaly`: Binary label (0 = normal, 1 = anomaly)

## Model Architecture

### Generator
- 3-layer LSTM architecture (32 → 64 → 128 hidden units)
- Output layer with Tanh activation

### Discriminator
- Single LSTM layer (100 hidden units)
- Sigmoid output for binary classification

## Configuration

You can modify training parameters in `run_tanogan.py`:

```python
class ArgsTrn:
    workers = 4          # Number of data loading workers
    batch_size = 32     # Batch size
    epochs = 20          # Number of training epochs
    lr = 0.0002          # Learning rate
    cuda = True          # Use GPU if available
    manualSeed = 2       # Random seed
```

## Results

After running the training script, results are saved in the `results/` directory:

- `training_history.csv`: Training loss history
- `anomaly_detection_results.csv`: Anomaly scores and predictions
- `performance_metrics.csv`: Evaluation metrics
- `anomaly_scores.npz`: Saved anomaly scores (for threshold testing)
- `models/generator.pth`: Trained generator weights
- `models/discriminator.pth`: Trained discriminator weights
- `images/`: Visualization plots

## Evaluation Metrics

The model reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Cohen's Kappa**: Agreement metric
- **ROC-AUC**: Area under the ROC curve

## Citation

If you use this code, please cite the original TAnoGAN paper:

```bibtex
@inproceedings{bashar2020tanogan,
  title={TAnoGAN: Time Series Anomaly Detection with Generative Adversarial Networks},
  author={Bashar, Md Abul and Nayak, Richi},
  booktitle={2020 IEEE Symposium Series on Computational Intelligence (SSCI)},
  pages={1778--1785},
  year={2020},
  organization={IEEE}
}
```

## License

[Add your license here]

## Contact

[Add your contact information here]
