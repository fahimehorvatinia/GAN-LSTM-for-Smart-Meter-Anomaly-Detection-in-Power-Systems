# GAN-LSTM for Smart Meter Anomaly Detection in Power Systems

This repository contains the implementation of a GAN-LSTM framework for anomaly detection in building-level electricity consumption data.

## Overview

This project implements a GAN-based approach for detecting anomalies in smart meter readings using the Large-scale Energy Anomaly Detection (LEAD) dataset. The model uses LSTM-based Generator and Discriminator networks to learn normal consumption patterns and identify anomalous behavior in power distribution systems.

## Features

- **LSTM-based GAN architecture** for time series anomaly detection
- **Adversarial temporal modeling** to capture nonlinear consumption patterns
- **Latent-space optimization** for anomaly scoring (TAnoGAN-style)
- **Comprehensive evaluation metrics** (Accuracy, Precision, Recall, F1-score, ROC-AUC)
- **Visualization tools** for anomaly detection results
- **Flexible threshold testing** without retraining

## Project Structure

```
energy-anomaly-detection-gan/
├── src/
│   ├── energy_dataset.py          # Dataset loader for LEAD energy data
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
git clone https://github.com/fahimehorvatinia/GAN-LSTM-for-Smart-Meter-Anomaly-Detection-in-Power-Systems.git
cd GAN-LSTM-for-Smart-Meter-Anomaly-Detection-in-Power-Systems
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place your `train.csv` file in the `data/` directory
   - The CSV should have columns: `timestamp`, `building_id`, `meter_reading`, and `anomaly` (for training data)
   - For the LEAD dataset format, see `data/README.md`

## Usage

### Training and Evaluation

Run the main script to train the model and evaluate on your data:

```bash
cd src
python run_tanogan.py
```

This will:
1. Load and preprocess the data (60-hour windowing, per-building normalization)
2. Train the GAN (Generator and Discriminator) on normal windows only
3. Calculate anomaly scores via latent-space optimization
4. Find the optimal threshold on validation data
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

The expected CSV format for training data (LEAD dataset format):

| timestamp | building_id | meter_reading | anomaly |
|-----------|-------------|---------------|---------|
| 2020-01-01 00:00:00 | 1 | 1234.5 | 0 |
| 2020-01-01 01:00:00 | 1 | 1235.2 | 0 |
| 2020-01-01 02:00:00 | 1 | 1500.0 | 1 |

- `timestamp`: DateTime string (hourly resolution)
- `building_id`: Building identifier
- `meter_reading`: Energy meter reading value (kWh)
- `anomaly`: Binary label (0 = normal, 1 = anomaly)

## Model Architecture

### Generator
- **Input**: Latent vector z ~ N(0, 0.1²) of dimension matching sequence length
- **Architecture**: 3-layer stacked LSTM (32 → 64 → 128 hidden units)
- **Output**: 60-step sequence with Tanh activation

### Discriminator
- **Input**: Normalized 60-step consumption sequence
- **Architecture**: Single LSTM layer (100 hidden units)
- **Output**: Sigmoid probability D(x) ∈ [0, 1]

### Anomaly Detection
- **Latent optimization**: TAnoGAN-style latent inversion (300 steps, Adam optimizer, lr=10⁻²)
- **Anomaly score**: S(x) = (1-λ)R(x) + λF(x), where λ=0.1
  - R(x): Residual loss (L1 distance)
  - F(x): Discriminator feature loss (L1 distance in feature space)

## Configuration

You can modify training parameters in `run_tanogan.py`:

```python
class ArgsTrn:
    workers = 4          # Number of data loading workers
    batch_size = 32     # Batch size
    epochs = 20          # Number of training epochs
    lr = 0.0002          # Learning rate (Adam)
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
- `images/`: Visualization plots (confusion matrix, ROC curve, time series plots)

## Evaluation Metrics

The model reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negatives / (True negatives + False positives)
- **ROC-AUC**: Area under the ROC curve

## Citation

Citation information will be added upon publication.

### Related Work

This implementation is inspired by and builds upon TAnoGAN:

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

## Dataset

This work uses the **Large-scale Energy Anomaly Detection (LEAD) dataset**:
- **Source**: [Kaggle - Large-scale Energy Anomaly Detection (LEAD)](https://www.kaggle.com/datasets/lead-dataset)
- **Size**: 406 buildings, 1 year of hourly data (8,760 hours per building)
- **Training**: 200 buildings
- **Testing**: 206 buildings
- **Total windows**: ~1.75M training windows, ~30K test windows (60-hour windows)

## Applications

This framework can support utilities and grid operators in:
- **Asset monitoring**: Detecting equipment degradation and malfunctions
- **Non-technical loss detection**: Identifying unauthorized energy usage
- **Situational awareness**: Enhancing real-time monitoring capabilities
- **Preventive maintenance**: Early detection of abnormal consumption patterns
- **Grid reliability**: Improving overall distribution network resilience

## Limitations and Future Work

- Current implementation focuses on univariate consumption data
- Future extensions: Multivariate inputs with weather, calendar, and building characteristics
- Improved handling of nonstationarity and regime shifts
- More efficient inference methods for real-time deployment

## License

[Add your license here]

## Contact

[Contact information will be added]

## Acknowledgments

We thank the LEAD dataset creators and the TAnoGAN authors for their foundational work.
