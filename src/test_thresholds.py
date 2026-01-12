#!/usr/bin/env python3
"""
TAnoGAN Threshold Testing Script

This script allows you to test different thresholds on a trained model
without needing to retrain or recalculate anomaly scores.

Usage:
    python test_thresholds.py [--threshold THRESHOLD] [--percentile PERCENTILE]
    
Examples:
    # Test with a specific threshold value
    python test_thresholds.py --threshold 4.0
    
    # Test with a specific percentile
    python test_thresholds.py --percentile 95
    
    # Interactive mode - test multiple thresholds
    python test_thresholds.py --interactive
"""

import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, roc_curve, auc, roc_auc_score, confusion_matrix
import seaborn as sns

# Import custom modules
from energy_dataset import EnergyDataset, EnergyDataSettings
from models.recurrent_models_pyramid import LSTMGenerator, LSTMDiscriminator

# Create results directory
RESULTS_DIR = 'results'
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
IMAGES_DIR = os.path.join(RESULTS_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

def load_models(device, in_dim=1):
    """Load saved generator and discriminator models"""
    generator_path = os.path.join(MODELS_DIR, 'generator.pth')
    discriminator_path = os.path.join(MODELS_DIR, 'discriminator.pth')
    
    if not os.path.exists(generator_path):
        raise FileNotFoundError(f"Generator weights not found at {generator_path}. Please run training first.")
    if not os.path.exists(discriminator_path):
        raise FileNotFoundError(f"Discriminator weights not found at {discriminator_path}. Please run training first.")
    
    # Create models
    netG = LSTMGenerator(in_dim=in_dim, out_dim=in_dim, device=device).to(device)
    netD = LSTMDiscriminator(in_dim=in_dim, device=device).to(device)
    
    # Load weights
    netG.load_state_dict(torch.load(generator_path, map_location=device))
    netD.load_state_dict(torch.load(discriminator_path, map_location=device))
    
    netG.eval()
    netD.eval()
    
    print(f"✓ Loaded generator from: {generator_path}")
    print(f"✓ Loaded discriminator from: {discriminator_path}")
    
    return netG, netD

def calculate_anomaly_scores(generator, discriminator, test_dataloader, device, window_length=60):
    """Calculate anomaly scores for test data"""
    def Anomaly_score(x, G_z, Lambda=0.1):
        residual_loss = torch.sum(torch.abs(x - G_z))
        output, x_feature = discriminator(x.to(device))
        output, G_z_feature = discriminator(G_z.to(device))
        discrimination_loss = torch.sum(torch.abs(x_feature - G_z_feature))
        total_loss = (1 - Lambda) * residual_loss.to(device) + Lambda * discrimination_loss
        return total_loss
    
    # Set models to training mode for backward pass (but we're not updating their weights)
    generator.train()
    discriminator.train()
    
    print("Calculating anomaly scores...")
    loss_list = []
    y_list = []
    
    total_samples = len(test_dataloader)
    for i, (x, y) in enumerate(test_dataloader):
        if (i + 1) % 1000 == 0:
            print(f"  Processing sample {i+1}/{total_samples}...")
        
        z = torch.nn.Parameter(torch.normal(0, 0.1, size=(1, window_length, 1)), requires_grad=True)
        z_optimizer = torch.optim.Adam([z], lr=1e-2)
        
        loss = None
        for j in range(50):
            z_optimizer.zero_grad()
            gen_fake, _ = generator(z.to(device))
            # x is already a tensor from the dataloader
            x_tensor = x.float().to(device) if isinstance(x, torch.Tensor) else torch.tensor(x).float().to(device)
            loss = Anomaly_score(x_tensor, gen_fake)
            loss.backward()
            z_optimizer.step()
        
        loss_list.append(loss.item() / window_length)
        y_list.append(y.item())
    
    # Set models back to eval mode
    generator.eval()
    discriminator.eval()
    
    print(f"✓ Completed anomaly score calculation for {len(loss_list)} samples")
    return np.array(loss_list), np.array(y_list)

def load_or_calculate_scores(generator, discriminator, test_dataloader, device, window_length=60):
    """Load saved anomaly scores or calculate them if not available"""
    scores_file = os.path.join(RESULTS_DIR, 'anomaly_scores.npz')
    
    if os.path.exists(scores_file):
        print(f"Loading saved anomaly scores from {scores_file}...")
        data = np.load(scores_file)
        loss_values = data['loss_values']
        y_values = data['y_values']
        print(f"✓ Loaded {len(loss_values)} anomaly scores")
        return loss_values, y_values
    else:
        print("Anomaly scores not found. Calculating...")
        loss_values, y_values = calculate_anomaly_scores(generator, discriminator, test_dataloader, device, window_length)
        
        # Save for future use
        np.savez(scores_file, loss_values=loss_values, y_values=y_values)
        print(f"✓ Saved anomaly scores to {scores_file}")
        return loss_values, y_values

def evaluate_threshold(loss_values, y_values, threshold):
    """Evaluate performance metrics for a given threshold"""
    predicted = (loss_values > threshold).astype(int)
    actual = y_values.astype(int)
    
    tp = np.count_nonzero(predicted * actual)
    tn = np.count_nonzero((predicted - 1) * (actual - 1))
    fp = np.count_nonzero(predicted * (actual - 1))
    fn = np.count_nonzero((predicted - 1) * actual)
    
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    kappa = cohen_kappa_score(actual, predicted)
    
    try:
        roc_auc = roc_auc_score(actual, loss_values)
    except:
        roc_auc = 0.0
    
    return {
        'threshold': threshold,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kappa': kappa,
        'roc_auc': roc_auc
    }

def print_metrics(metrics):
    """Print formatted metrics"""
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print(f"Threshold: {metrics['threshold']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positive:  {metrics['tp']:6d}")
    print(f"  True Negative:  {metrics['tn']:6d}")
    print(f"  False Positive: {metrics['fp']:6d}")
    print(f"  False Negative: {metrics['fn']:6d}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  Kappa:     {metrics['kappa']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 80)

def plot_results(loss_values, y_values, threshold, save_path=None):
    """Plot anomaly scores with threshold"""
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(range(len(loss_values)), loss_values, label='Anomaly Score', alpha=0.7)
    ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    ax.plot(range(len(loss_values)), y_values * threshold, label='Ground Truth Anomaly', alpha=0.5)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Anomaly Detection Results')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to: {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Test different thresholds on trained TAnoGAN model')
    parser.add_argument('--threshold', type=float, help='Specific threshold value to test')
    parser.add_argument('--percentile', type=float, help='Percentile to use as threshold (0-100)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode to test multiple thresholds')
    # Get default data file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_data_file = os.path.join(project_root, 'data', 'train.csv')
    parser.add_argument('--data-file', type=str, default=default_data_file, help='Path to data file')
    parser.add_argument('--window-length', type=int, default=60, help='Window length for sequences')
    parser.add_argument('--save-results', action='store_true', help='Save results to CSV')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    print("\nLoading models...")
    generator, discriminator = load_models(device, in_dim=1)
    
    # Load test data
    print("\nLoading test data...")
    test_data_settings = EnergyDataSettings()
    test_data_settings.data_file = args.data_file
    test_data_settings.train = False
    test_data_settings.window_length = args.window_length
    test_data_settings.group_by_building = False
    
    test_dataset = EnergyDataset(test_data_settings)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load or calculate anomaly scores
    print("\nLoading/calculating anomaly scores...")
    loss_values, y_values = load_or_calculate_scores(generator, discriminator, test_dataloader, device, args.window_length)
    
    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 80)
        print("INTERACTIVE MODE")
        print("=" * 80)
        print("Enter threshold values to test (or 'q' to quit, 'p' for percentile)")
        print("Examples:")
        print("  4.0        - Test with threshold 4.0")
        print("  p95        - Test with 95th percentile")
        print("  best       - Find best threshold")
        print("=" * 80)
        
        while True:
            try:
                user_input = input("\nEnter threshold (or 'q' to quit): ").strip().lower()
                
                if user_input == 'q':
                    break
                elif user_input == 'best':
                    # Find best threshold
                    print("Searching for best threshold...")
                    best_metrics = None
                    best_threshold = None
                    best_accuracy = 0
                    
                    for percentile in range(50, 100, 1):
                        threshold = np.percentile(loss_values, percentile)
                        metrics = evaluate_threshold(loss_values, y_values, threshold)
                        if metrics['accuracy'] > best_accuracy:
                            best_accuracy = metrics['accuracy']
                            best_threshold = threshold
                            best_metrics = metrics
                    
                    print_metrics(best_metrics)
                    plot_results(loss_values, y_values, best_threshold, 
                               os.path.join(IMAGES_DIR, f'anomaly_scores_threshold_{best_threshold:.4f}.png'))
                    
                elif user_input.startswith('p'):
                    # Percentile mode
                    percentile = float(user_input[1:])
                    threshold = np.percentile(loss_values, percentile)
                    print(f"Using {percentile}th percentile: {threshold:.4f}")
                    metrics = evaluate_threshold(loss_values, y_values, threshold)
                    print_metrics(metrics)
                    plot_results(loss_values, y_values, threshold,
                               os.path.join(IMAGES_DIR, f'anomaly_scores_threshold_{threshold:.4f}.png'))
                    
                else:
                    # Direct threshold value
                    threshold = float(user_input)
                    metrics = evaluate_threshold(loss_values, y_values, threshold)
                    print_metrics(metrics)
                    plot_results(loss_values, y_values, threshold,
                               os.path.join(IMAGES_DIR, f'anomaly_scores_threshold_{threshold:.4f}.png'))
                    
            except ValueError:
                print("Invalid input. Please enter a number, 'p' followed by percentile, 'best', or 'q'.")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    # Single threshold test
    else:
        if args.threshold is not None:
            threshold = args.threshold
        elif args.percentile is not None:
            threshold = np.percentile(loss_values, args.percentile)
            print(f"Using {args.percentile}th percentile: {threshold:.4f}")
        else:
            # Default: find best threshold
            print("Finding best threshold...")
            best_metrics = None
            best_threshold = None
            best_accuracy = 0
            
            for percentile in range(50, 100, 1):
                test_threshold = np.percentile(loss_values, percentile)
                metrics = evaluate_threshold(loss_values, y_values, test_threshold)
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_threshold = test_threshold
                    best_metrics = metrics
            
            threshold = best_threshold
            print(f"Best threshold found: {threshold:.4f}")
        
        # Evaluate
        metrics = evaluate_threshold(loss_values, y_values, threshold)
        print_metrics(metrics)
        
        # Plot
        plot_results(loss_values, y_values, threshold,
                    os.path.join(IMAGES_DIR, f'anomaly_scores_threshold_{threshold:.4f}.png'))
        
        # Save results if requested
        if args.save_results:
            results_df = pd.DataFrame({
                'loss': loss_values,
                'y': y_values,
                'threshold': threshold,
                'anomaly': (loss_values > threshold).astype(int)
            })
            results_file = os.path.join(RESULTS_DIR, f'anomaly_detection_results_threshold_{threshold:.4f}.csv')
            results_df.to_csv(results_file, index=False)
            print(f"\n✓ Saved results to: {results_file}")
            
            # Save metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_file = os.path.join(RESULTS_DIR, f'performance_metrics_threshold_{threshold:.4f}.csv')
            metrics_df.to_csv(metrics_file, index=False)
            print(f"✓ Saved metrics to: {metrics_file}")

if __name__ == '__main__':
    main()

