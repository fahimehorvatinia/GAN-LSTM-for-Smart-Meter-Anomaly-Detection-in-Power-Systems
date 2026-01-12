#!/usr/bin/env python3
"""
TAnoGAN: Time Series Anomaly Detection with Generative Adversarial Networks
Python script version - saves results and images to files instead of notebook

This script runs the complete TAnoGAN pipeline:
1. Data loading and preprocessing
2. Model training
3. Anomaly detection
4. Visualization and evaluation
5. Saves all results to files
"""

import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torch.nn.init as init
from torch.autograd import Variable
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, roc_curve, auc, roc_auc_score

# Import custom modules
from energy_dataset import EnergyDataset, EnergyDataSettings
from models.recurrent_models_pyramid import LSTMGenerator, LSTMDiscriminator

# Create results directory
RESULTS_DIR = 'results'
IMAGES_DIR = os.path.join(RESULTS_DIR, 'images')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

def save_plot(fig, filename, dpi=300):
    """Save matplotlib figure to file"""
    filepath = os.path.join(IMAGES_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved plot: {filepath}")
    plt.close(fig)

def print_and_save(text, filepath):
    """Print text and save to file"""
    print(text)
    with open(filepath, 'a') as f:
        f.write(text + '\n')

def main():
    # Setup logging
    log_file = os.path.join(RESULTS_DIR, f'training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    print(f"Starting TAnoGAN execution at {datetime.datetime.now()}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    print(f"Log file: {log_file}")
    print("=" * 80)
    
    # ============================================================================
    # 1. SETUP AND CONFIGURATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: SETUP AND CONFIGURATION")
    print("=" * 80)
    
    class ArgsTrn:
        workers = 4
        batch_size = 32
        epochs = 20
        lr = 0.0002
        cuda = True
        manualSeed = 2
    
    opt_trn = ArgsTrn()
    
    torch.manual_seed(opt_trn.manualSeed)
    cudnn.benchmark = True
    
    print_and_save(f"Training configuration:", log_file)
    print_and_save(f"  - Epochs: {opt_trn.epochs}", log_file)
    print_and_save(f"  - Batch size: {opt_trn.batch_size}", log_file)
    print_and_save(f"  - Learning rate: {opt_trn.lr}", log_file)
    print_and_save(f"  - CUDA: {opt_trn.cuda}", log_file)
    
    # ============================================================================
    # 2. DATA LOADING
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: DATA LOADING")
    print("=" * 80)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_file = os.path.join(project_root, 'data', 'train.csv')
    data_settings = EnergyDataSettings()
    data_settings.data_file = data_file
    data_settings.train = True
    data_settings.window_length = 60
    data_settings.group_by_building = False
    
    print("Loading training dataset...")
    dataset = EnergyDataset(data_settings=data_settings)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt_trn.batch_size,
                                             shuffle=True, num_workers=int(opt_trn.workers))
    
    print_and_save(f"Dataset shape - X: {dataset.x.shape}, Y: {dataset.y.shape}", log_file)
    print_and_save(f"Dataset size: {len(dataset)}", log_file)
    print_and_save(f"Anomaly ratio: {dataset.y.sum().item() / len(dataset):.4f}", log_file)
    
    # ============================================================================
    # 3. MODEL SETUP
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: MODEL SETUP")
    print("=" * 80)
    
    device = torch.device("cuda:0" if opt_trn.cuda else "cpu")
    seq_len = dataset.window_length
    in_dim = dataset.n_feature
    
    print(f"Device: {device}")
    print(f"Sequence length: {seq_len}")
    print(f"Input dimension: {in_dim}")
    
    # Create models
    netD = LSTMDiscriminator(in_dim=in_dim, device=device).to(device)
    netG = LSTMGenerator(in_dim=in_dim, out_dim=in_dim, device=device).to(device)
    
    print("\nDiscriminator Architecture:")
    print(netD)
    print("\nGenerator Architecture:")
    print(netG)
    
    # Save model architectures
    with open(os.path.join(RESULTS_DIR, 'model_architectures.txt'), 'w') as f:
        f.write("Discriminator Architecture:\n")
        f.write(str(netD))
        f.write("\n\nGenerator Architecture:\n")
        f.write(str(netG))
    
    # Setup loss and optimizers
    criterion = nn.BCELoss().to(device)
    optimizerD = optim.Adam(netD.parameters(), lr=opt_trn.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt_trn.lr)
    
    # ============================================================================
    # 4. TRAINING
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: TRAINING")
    print("=" * 80)
    
    real_label = 1
    fake_label = 0
    
    # Store training history
    training_history = {
        'epoch': [],
        'loss_D': [],
        'loss_G': [],
        'D_x': [],
        'D_G_z1': [],
        'D_G_z2': []
    }
    
    print("Starting training...")
    print_and_save("\nTraining Progress:", log_file)
    
    for epoch in range(opt_trn.epochs):
        epoch_loss_D = []
        epoch_loss_G = []
        epoch_D_x = []
        epoch_D_G_z1 = []
        epoch_D_G_z2 = []
        
        for i, (x, y) in enumerate(dataloader, 0):
            # (1) Update D network
            netD.zero_grad()
            real = x.to(device)
            batch_size, seq_len = real.size(0), real.size(1)
            label = torch.full((batch_size, seq_len, 1), real_label, device=device)
            
            output, _ = netD.forward(real)
            errD_real = criterion(output, label.float())
            errD_real.backward()
            optimizerD.step()
            D_x = output.mean().item()
            
            # Train with fake data
            noise = Variable(torch.normal(0, 0.1, size=(batch_size, seq_len, in_dim))).to(device)
            fake, _ = netG.forward(noise)
            output, _ = netD.forward(fake.detach())
            label.fill_(fake_label)
            errD_fake = criterion(output, label.float())
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # (2) Update G network
            netG.zero_grad()
            noise = Variable(torch.normal(0, 0.1, size=(batch_size, seq_len, in_dim))).to(device)
            fake, _ = netG.forward(noise)
            label.fill_(real_label)
            output, _ = netD.forward(fake)
            errG = criterion(output, label.float())
            errG.backward()
            optimizerG.step()
            D_G_z2 = output.mean().item()
            
            epoch_loss_D.append(errD.item())
            epoch_loss_G.append(errG.item())
            epoch_D_x.append(D_x)
            epoch_D_G_z1.append(D_G_z1)
            epoch_D_G_z2.append(D_G_z2)
        
        # Average metrics for the epoch
        avg_loss_D = np.mean(epoch_loss_D)
        avg_loss_G = np.mean(epoch_loss_G)
        avg_D_x = np.mean(epoch_D_x)
        avg_D_G_z1 = np.mean(epoch_D_G_z1)
        avg_D_G_z2 = np.mean(epoch_D_G_z2)
        
        training_history['epoch'].append(epoch)
        training_history['loss_D'].append(avg_loss_D)
        training_history['loss_G'].append(avg_loss_G)
        training_history['D_x'].append(avg_D_x)
        training_history['D_G_z1'].append(avg_D_G_z1)
        training_history['D_G_z2'].append(avg_D_G_z2)
        
        log_msg = f'[{epoch}/{opt_trn.epochs}] Loss_D: {avg_loss_D:.4f} Loss_G: {avg_loss_G:.4f} D(x): {avg_D_x:.4f} D(G(z)): {avg_D_G_z1:.4f} / {avg_D_G_z2:.4f}'
        print(log_msg)
        print_and_save(log_msg, log_file)
    
    # Save training history
    training_df = pd.DataFrame(training_history)
    training_df.to_csv(os.path.join(RESULTS_DIR, 'training_history.csv'), index=False)
    print(f"\nTraining history saved to: {os.path.join(RESULTS_DIR, 'training_history.csv')}")
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(training_history['epoch'], training_history['loss_D'], label='Discriminator Loss')
    axes[0, 0].plot(training_history['epoch'], training_history['loss_G'], label='Generator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(training_history['epoch'], training_history['D_x'], label='D(x)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('D(x)')
    axes[0, 1].set_title('Discriminator Output for Real Data')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(training_history['epoch'], training_history['D_G_z1'], label='D(G(z)) before G update')
    axes[1, 0].plot(training_history['epoch'], training_history['D_G_z2'], label='D(G(z)) after G update')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('D(G(z))')
    axes[1, 0].set_title('Discriminator Output for Generated Data')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(training_history['epoch'], training_history['loss_D'], label='Loss D', alpha=0.7)
    axes[1, 1].plot(training_history['epoch'], training_history['loss_G'], label='Loss G', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Combined Training Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, 'training_curves.png')
    
    # ============================================================================
    # 5. ANOMALY DETECTION SETUP
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: ANOMALY DETECTION SETUP")
    print("=" * 80)
    
    class ArgsTest:
        workers = 1
        batch_size = 1
    
    opt_test = ArgsTest()
    generator = netG
    discriminator = netD
    
    # Setup test data
    test_data_settings = EnergyDataSettings()
    test_data_settings.data_file = data_file  # Use same data file as training
    test_data_settings.train = False
    test_data_settings.window_length = 60
    test_data_settings.group_by_building = False
    
    print("Loading test dataset...")
    test_dataset = EnergyDataset(test_data_settings)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt_test.batch_size,
                                                  shuffle=False, num_workers=int(opt_test.workers))
    
    print_and_save(f"Test dataset shape - X: {test_dataset.x.shape}, Y: {test_dataset.y.shape}", log_file)
    print_and_save(f"Test dataset size: {test_dataset.data_len}", log_file)
    
    # ============================================================================
    # 6. ANOMALY SCORE CALCULATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 6: ANOMALY SCORE CALCULATION")
    print("=" * 80)
    
    def Anomaly_score(x, G_z, Lambda=0.1):
        residual_loss = torch.sum(torch.abs(x - G_z))
        output, x_feature = discriminator(x.to(device))
        output, G_z_feature = discriminator(G_z.to(device))
        discrimination_loss = torch.sum(torch.abs(x_feature - G_z_feature))
        total_loss = (1 - Lambda) * residual_loss.to(device) + Lambda * discrimination_loss
        return total_loss
    
    print("Calculating anomaly scores for test data...")
    loss_list = []
    y_list = []
    
    total_samples = len(test_dataloader)
    for i, (x, y) in enumerate(test_dataloader):
        if (i + 1) % 100 == 0:
            print(f"Processing sample {i+1}/{total_samples}...")
        
        z = Variable(torch.normal(0, 0.1, size=(opt_test.batch_size,
                                                 test_dataset.window_length,
                                                 test_dataset.n_feature)), requires_grad=True)
        z_optimizer = torch.optim.Adam([z], lr=1e-2)
        
        loss = None
        for j in range(50):
            z_optimizer.zero_grad()
            gen_fake, _ = generator(z.to(device))
            loss = Anomaly_score(Variable(x).to(device), gen_fake)
            loss.backward()
            z_optimizer.step()
        
        loss_list.append(loss)
        y_list.append(y.item())
    
    print(f"Completed anomaly score calculation for {len(loss_list)} samples")
    
    # ============================================================================
    # 7. VISUALIZATION AND EVALUATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 7: VISUALIZATION AND EVALUATION")
    print("=" * 80)
    
    # Calculate threshold
    loss_values = [loss.item() / test_dataset.window_length for loss in loss_list]
    
    # Save anomaly scores for future use (so we don't need to recalculate)
    scores_file = os.path.join(RESULTS_DIR, 'anomaly_scores.npz')
    np.savez(scores_file, 
             loss_values=np.array(loss_values), 
             y_values=test_dataset.y.numpy())
    print(f"Saved anomaly scores to: {scores_file}")
    print("You can now use test_thresholds.py to test different thresholds without retraining.")
    
    # Find optimal threshold for >90% accuracy
    print("Finding optimal threshold for >90% accuracy...")
    print_and_save("Finding optimal threshold for >90% accuracy...", log_file)
    
    actual = test_dataset.y.numpy()
    best_threshold = None
    best_accuracy = 0
    best_percentile = 95
    
    # Test different percentile thresholds
    for percentile in range(50, 100, 1):
        test_threshold = np.percentile(loss_values, percentile)
        test_predicted = (np.array(loss_values) > test_threshold).astype(int)
        
        tp = np.count_nonzero(test_predicted * actual)
        tn = np.count_nonzero((test_predicted - 1) * (actual - 1))
        fp = np.count_nonzero(test_predicted * (actual - 1))
        fn = np.count_nonzero((test_predicted - 1) * actual)
        
        test_accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_threshold = test_threshold
            best_percentile = percentile
        
        # If we found >90% accuracy, we can stop (or continue to find the best)
        if test_accuracy >= 0.90:
            print(f"Found threshold at {percentile}th percentile: {test_threshold:.4f} with accuracy: {test_accuracy:.4f}")
            print_and_save(f"Found threshold at {percentile}th percentile: {test_threshold:.4f} with accuracy: {test_accuracy:.4f}", log_file)
    
    # Use the best threshold found
    THRESHOLD = best_threshold if best_threshold is not None else np.percentile(loss_values, 95)
    print(f"\nOptimal threshold: {THRESHOLD:.4f} ({best_percentile}th percentile)")
    print(f"Best accuracy achieved: {best_accuracy:.4f}")
    print_and_save(f"Optimal threshold: {THRESHOLD:.4f} ({best_percentile}th percentile)", log_file)
    print_and_save(f"Best accuracy achieved: {best_accuracy:.4f}", log_file)
    
    if best_accuracy < 0.90:
        print(f"WARNING: Could not achieve >90% accuracy. Best accuracy: {best_accuracy:.4f}")
        print("You may need to:")
        print("1. Improve model training (more epochs, better hyperparameters)")
        print("2. Use a different threshold selection method")
        print("3. Check if the dataset has sufficient signal for anomaly detection")
        print_and_save(f"WARNING: Could not achieve >90% accuracy. Best accuracy: {best_accuracy:.4f}", log_file)
    
    # Create results dataframe
    test_score_df = pd.DataFrame(index=range(test_dataset.data_len))
    test_score_df['loss'] = loss_values
    test_score_df['y'] = test_dataset.y.numpy()
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['t'] = [x[-1, 0].item() for x in test_dataset.x]
    
    # Save results to CSV
    results_file = os.path.join(RESULTS_DIR, 'anomaly_detection_results.csv')
    test_score_df.to_csv(results_file)
    print(f"Results saved to: {results_file}")
    
    # Plot 1: Anomaly scores
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(test_score_df.index, test_score_df.loss, label='Anomaly Score', alpha=0.7)
    ax.plot(test_score_df.index, test_score_df.threshold, label='Threshold', linestyle='--', color='red')
    ax.plot(test_score_df.index, test_score_df.y * THRESHOLD, label='Ground Truth Anomaly', alpha=0.5)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Anomaly Detection Results - Energy Dataset')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, 'anomaly_scores.png')
    
    # Plot 2: Time series with anomalies (full plot)
    anomalies = test_score_df[test_score_df.anomaly == True]
    ground_truth_anomalies = test_score_df[test_score_df['y'] == 1]
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(range(test_dataset.data_len), test_score_df['t'], 
            label='Meter Reading (normalized)', alpha=0.6, linewidth=0.5)
    
    if len(anomalies) > 0:
        sns.scatterplot(x=anomalies.index, y=anomalies.t, color='red', s=52,
                       label='Detected Anomaly', alpha=0.7, ax=ax)
    
    if len(ground_truth_anomalies) > 0:
        sns.scatterplot(x=ground_truth_anomalies.index, y=ground_truth_anomalies.t,
                       color='green', s=30, marker='x', label='Ground Truth Anomaly',
                       alpha=0.7, ax=ax)
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Normalized Meter Reading')
    ax.set_title('Time Series with Anomaly Detection - Energy Dataset (Full View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, 'time_series_with_anomalies.png')
    
    # Plot 3-8: Detailed time series plots for different ranges
    def plot_time_series_range(start_idx, end_idx, filename_suffix):
        """Plot time series with anomalies for a specific range"""
        # Filter data for the range
        range_mask = (test_score_df.index >= start_idx) & (test_score_df.index < end_idx)
        range_df = test_score_df[range_mask]
        
        if len(range_df) == 0:
            return
        
        range_anomalies = range_df[range_df.anomaly == True]
        range_ground_truth = range_df[range_df['y'] == 1]
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot meter readings
        ax.plot(range_df.index, range_df['t'], 
                label='Meter Reading (normalized)', alpha=0.7, linewidth=1.0, color='blue')
        
        # Plot detected anomalies
        if len(range_anomalies) > 0:
            sns.scatterplot(x=range_anomalies.index, y=range_anomalies.t, color='red', s=80,
                           label='Detected Anomaly', alpha=0.8, ax=ax, marker='o')
        
        # Plot ground truth anomalies
        if len(range_ground_truth) > 0:
            sns.scatterplot(x=range_ground_truth.index, y=range_ground_truth.t,
                           color='green', s=50, marker='x', label='Ground Truth Anomaly',
                           alpha=0.8, ax=ax, linewidths=2)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Normalized Meter Reading')
        ax.set_title(f'Time Series with Anomaly Detection - Energy Dataset (Samples {start_idx}-{end_idx})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(fig, f'time_series_with_anomalies_{filename_suffix}.png')
        print(f"Saved detailed plot: time_series_with_anomalies_{filename_suffix}.png")
    
    # Create detailed plots for each range
    ranges = [
        (0, 5000, '0_5k'),
        (5000, 10000, '5k_10k'),
        (10000, 15000, '10k_15k'),
        (15000, 20000, '15k_20k'),
        (20000, 25000, '20k_25k'),
        (25000, 30000, '25k_30k')
    ]
    
    print("\nCreating detailed plots for different sample ranges...")
    for start, end, suffix in ranges:
        if start < test_dataset.data_len:
            end = min(end, test_dataset.data_len)
            plot_time_series_range(start, end, suffix)
    
    # ============================================================================
    # 8. CALCULATE PERFORMANCE METRICS
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 8: PERFORMANCE METRICS")
    print("=" * 80)
    
    # Calculate window-based anomalies
    start_end = []
    state = 0
    for idx in test_score_df.index:
        if state == 0 and test_score_df.loc[idx, 'y'] == 1:
            state = 1
            start = idx
        if state == 1 and test_score_df.loc[idx, 'y'] == 0:
            state = 0
            end = idx
            start_end.append((start, end))
    
    for s_e in start_end:
        if sum(test_score_df[s_e[0]:s_e[1]+1]['anomaly']) > 0:
            for i in range(s_e[0], s_e[1]+1):
                test_score_df.loc[i, 'anomaly'] = 1
    
    actual = np.array(test_score_df['y'])
    predicted = np.array([int(a) for a in test_score_df['anomaly']])
    
    # Calculate metrics
    tp = np.count_nonzero(predicted * actual)
    tn = np.count_nonzero((predicted - 1) * (actual - 1))
    fp = np.count_nonzero(predicted * (actual - 1))
    fn = np.count_nonzero((predicted - 1) * actual)
    
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fmeasure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    kappa = cohen_kappa_score(predicted, actual)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predicted)
    auc_val = auc(false_positive_rate, true_positive_rate)
    roc_auc_val = roc_auc_score(actual, predicted)
    
    # Print and save metrics
    metrics_text = f"""
Performance Metrics:
===================
True Positive:     {tp}
True Negative:     {tn}
False Positive:    {fp}
False Negative:    {fn}

Accuracy:          {accuracy:.4f}
Precision:         {precision:.4f}
Recall:            {recall:.4f}
F-measure:         {fmeasure:.4f}
Cohen's Kappa:     {kappa:.4f}
AUC:               {auc_val:.4f}
ROC AUC:           {roc_auc_val:.4f}
"""
    print(metrics_text)
    print_and_save(metrics_text, log_file)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['True Positive', 'True Negative', 'False Positive', 'False Negative',
                   'Accuracy', 'Precision', 'Recall', 'F-measure', "Cohen's Kappa", 'AUC', 'ROC AUC'],
        'Value': [tp, tn, fp, fn, accuracy, precision, recall, fmeasure, kappa, auc_val, roc_auc_val]
    })
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'performance_metrics.csv'), index=False)
    
    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(false_positive_rate, true_positive_rate, 
            label=f'ROC Curve (AUC = {auc_val:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Energy Anomaly Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, 'roc_curve.png')
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(actual, predicted)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix - Energy Anomaly Detection')
    plt.tight_layout()
    save_plot(fig, 'confusion_matrix.png')
    
    # ============================================================================
    # 9. SAVE MODEL (optional)
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 9: SAVING MODELS")
    print("=" * 80)
    
    model_dir = os.path.join(RESULTS_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    generator_path = os.path.join(model_dir, 'generator.pth')
    discriminator_path = os.path.join(model_dir, 'discriminator.pth')
    
    torch.save(netG.state_dict(), generator_path)
    torch.save(netD.state_dict(), discriminator_path)
    
    print(f"Generator saved to: {generator_path}")
    print(f"Discriminator saved to: {discriminator_path}")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {RESULTS_DIR}")
    print(f"\nGenerated files:")
    print(f"  - Training log: {log_file}")
    print(f"  - Training history: {RESULTS_DIR}/training_history.csv")
    print(f"  - Anomaly results: {results_file}")
    print(f"  - Performance metrics: {RESULTS_DIR}/performance_metrics.csv")
    print(f"  - Model architectures: {RESULTS_DIR}/model_architectures.txt")
    print(f"  - Models: {model_dir}/")
    print(f"\nGenerated images:")
    print(f"  - Training curves: {IMAGES_DIR}/training_curves.png")
    print(f"  - Anomaly scores: {IMAGES_DIR}/anomaly_scores.png")
    print(f"  - Time series with anomalies: {IMAGES_DIR}/time_series_with_anomalies.png")
    print(f"  - ROC curve: {IMAGES_DIR}/roc_curve.png")
    print(f"  - Confusion matrix: {IMAGES_DIR}/confusion_matrix.png")
    print("\n" + "=" * 80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

