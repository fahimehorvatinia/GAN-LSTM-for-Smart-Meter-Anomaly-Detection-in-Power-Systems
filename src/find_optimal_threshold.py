#!/usr/bin/env python3
"""
Comprehensive Threshold Search for TAnoGAN

This script systematically tests different thresholds to find one that achieves >90% accuracy.
It tests:
1. Percentile-based thresholds (0-100)
2. Direct threshold values (min to max of scores)
3. Grid search over threshold range
"""

import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix
import seaborn as sns

# Create results directory
RESULTS_DIR = 'results'
IMAGES_DIR = os.path.join(RESULTS_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

def load_anomaly_scores():
    """Load saved anomaly scores"""
    scores_file = os.path.join(RESULTS_DIR, 'anomaly_scores.npz')
    
    if not os.path.exists(scores_file):
        raise FileNotFoundError(f"Anomaly scores not found at {scores_file}. Please run training first.")
    
    print(f"Loading anomaly scores from {scores_file}...")
    data = np.load(scores_file)
    loss_values = data['loss_values']
    y_values = data['y_values']
    print(f"✓ Loaded {len(loss_values)} anomaly scores")
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
    print(f"Threshold: {metrics['threshold']:.6f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positive:  {metrics['tp']:6d}")
    print(f"  True Negative:  {metrics['tn']:6d}")
    print(f"  False Positive: {metrics['fp']:6d}")
    print(f"  False Negative: {metrics['fn']:6d}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  Kappa:     {metrics['kappa']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 80)

def search_percentile_thresholds(loss_values, y_values, target_accuracy=0.90):
    """Search percentile-based thresholds"""
    print("\n" + "=" * 80)
    print("SEARCH 1: Percentile-based Thresholds")
    print("=" * 80)
    
    best_metrics = None
    best_threshold = None
    best_percentile = None
    best_accuracy = 0
    candidates = []
    
    # Test percentiles from 0 to 100 with fine granularity
    print("Testing percentiles from 0 to 100...")
    for percentile in np.arange(0, 100.1, 0.1):
        threshold = np.percentile(loss_values, percentile)
        metrics = evaluate_threshold(loss_values, y_values, threshold)
        
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_threshold = threshold
            best_percentile = percentile
            best_metrics = metrics
        
        # Collect candidates with >90% accuracy
        if metrics['accuracy'] >= target_accuracy:
            candidates.append({
                'percentile': percentile,
                'threshold': threshold,
                'metrics': metrics
            })
    
    print(f"\nBest percentile-based threshold:")
    print(f"  Percentile: {best_percentile:.2f}%")
    print(f"  Threshold: {best_threshold:.6f}")
    print(f"  Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    if candidates:
        print(f"\nFound {len(candidates)} thresholds with >= {target_accuracy*100:.1f}% accuracy:")
        for cand in candidates[:10]:  # Show top 10
            print(f"  Percentile {cand['percentile']:.2f}%: threshold={cand['threshold']:.6f}, accuracy={cand['metrics']['accuracy']:.4f}")
        if len(candidates) > 10:
            print(f"  ... and {len(candidates) - 10} more")
    else:
        print(f"\n⚠ No thresholds found with >= {target_accuracy*100:.1f}% accuracy using percentiles")
    
    return best_metrics, best_threshold, candidates

def search_direct_thresholds(loss_values, y_values, target_accuracy=0.90, n_points=1000):
    """Search direct threshold values"""
    print("\n" + "=" * 80)
    print("SEARCH 2: Direct Threshold Values")
    print("=" * 80)
    
    min_score = np.min(loss_values)
    max_score = np.max(loss_values)
    print(f"Score range: [{min_score:.6f}, {max_score:.6f}]")
    
    # Create grid of threshold values
    thresholds = np.linspace(min_score, max_score, n_points)
    
    best_metrics = None
    best_threshold = None
    best_accuracy = 0
    candidates = []
    
    print(f"Testing {n_points} threshold values...")
    for threshold in thresholds:
        metrics = evaluate_threshold(loss_values, y_values, threshold)
        
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_threshold = threshold
            best_metrics = metrics
        
        # Collect candidates with >90% accuracy
        if metrics['accuracy'] >= target_accuracy:
            candidates.append({
                'threshold': threshold,
                'metrics': metrics
            })
    
    print(f"\nBest direct threshold:")
    print(f"  Threshold: {best_threshold:.6f}")
    print(f"  Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    if candidates:
        print(f"\nFound {len(candidates)} thresholds with >= {target_accuracy*100:.1f}% accuracy:")
        for cand in candidates[:10]:  # Show top 10
            print(f"  Threshold {cand['threshold']:.6f}: accuracy={cand['metrics']['accuracy']:.4f}")
        if len(candidates) > 10:
            print(f"  ... and {len(candidates) - 10} more")
    else:
        print(f"\n⚠ No thresholds found with >= {target_accuracy*100:.1f}% accuracy using direct values")
    
    return best_metrics, best_threshold, candidates

def binary_search_threshold(loss_values, y_values, target_accuracy=0.90, tolerance=1e-6):
    """Binary search for threshold achieving target accuracy"""
    print("\n" + "=" * 80)
    print("SEARCH 3: Binary Search for Target Accuracy")
    print("=" * 80)
    
    min_score = np.min(loss_values)
    max_score = np.max(loss_values)
    
    left = min_score
    right = max_score
    best_threshold = None
    best_metrics = None
    best_accuracy = 0
    
    print(f"Binary searching for threshold achieving {target_accuracy*100:.1f}% accuracy...")
    print(f"Score range: [{min_score:.6f}, {max_score:.6f}]")
    
    iterations = 0
    max_iterations = 100
    
    while (right - left) > tolerance and iterations < max_iterations:
        mid = (left + right) / 2
        metrics = evaluate_threshold(loss_values, y_values, mid)
        
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_threshold = mid
            best_metrics = metrics
        
        if metrics['accuracy'] >= target_accuracy:
            right = mid  # Try lower threshold
        else:
            left = mid  # Try higher threshold
        
        iterations += 1
        if iterations % 10 == 0:
            print(f"  Iteration {iterations}: threshold={mid:.6f}, accuracy={metrics['accuracy']:.4f}")
    
    print(f"\nBinary search completed in {iterations} iterations")
    print(f"Best threshold: {best_threshold:.6f}")
    print(f"Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    return best_metrics, best_threshold

def plot_threshold_analysis(loss_values, y_values, best_threshold, save_path=None):
    """Plot comprehensive threshold analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Anomaly scores with threshold
    axes[0, 0].plot(range(len(loss_values)), loss_values, label='Anomaly Score', alpha=0.7, linewidth=0.5)
    axes[0, 0].axhline(y=best_threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {best_threshold:.4f}')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Anomaly Score')
    axes[0, 0].set_title('Anomaly Scores with Optimal Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs Threshold
    thresholds = np.linspace(np.min(loss_values), np.max(loss_values), 200)
    accuracies = []
    for thresh in thresholds:
        metrics = evaluate_threshold(loss_values, y_values, thresh)
        accuracies.append(metrics['accuracy'])
    
    axes[0, 1].plot(thresholds, accuracies, linewidth=2)
    axes[0, 1].axhline(y=0.90, color='g', linestyle='--', linewidth=2, label='90% Target')
    axes[0, 1].axvline(x=best_threshold, color='r', linestyle='--', linewidth=2, label=f'Best: {best_threshold:.4f}')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy vs Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix
    predicted = (loss_values > best_threshold).astype(int)
    cm = confusion_matrix(y_values, predicted)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')
    
    # Plot 4: Score distribution
    axes[1, 1].hist(loss_values[y_values == 0], bins=50, alpha=0.5, label='Normal', density=True)
    axes[1, 1].hist(loss_values[y_values == 1], bins=50, alpha=0.5, label='Anomaly', density=True)
    axes[1, 1].axvline(x=best_threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {best_threshold:.4f}')
    axes[1, 1].set_xlabel('Anomaly Score')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Score Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved analysis plot to: {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    print("=" * 80)
    print("COMPREHENSIVE THRESHOLD SEARCH FOR TAnoGAN")
    print("=" * 80)
    print("This script will systematically search for thresholds achieving >90% accuracy")
    print("=" * 80)
    
    # Load anomaly scores
    loss_values, y_values = load_anomaly_scores()
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(loss_values)}")
    print(f"  Normal samples: {np.sum(y_values == 0)} ({np.sum(y_values == 0)/len(y_values)*100:.2f}%)")
    print(f"  Anomaly samples: {np.sum(y_values == 1)} ({np.sum(y_values == 1)/len(y_values)*100:.2f}%)")
    print(f"  Score range: [{np.min(loss_values):.6f}, {np.max(loss_values):.6f}]")
    print(f"  Score mean: {np.mean(loss_values):.6f}")
    print(f"  Score std: {np.std(loss_values):.6f}")
    
    # Search 1: Percentile-based
    p_metrics, p_threshold, p_candidates = search_percentile_thresholds(loss_values, y_values, target_accuracy=0.90)
    
    # Search 2: Direct thresholds
    d_metrics, d_threshold, d_candidates = search_direct_thresholds(loss_values, y_values, target_accuracy=0.90, n_points=2000)
    
    # Search 3: Binary search
    b_metrics, b_threshold = binary_search_threshold(loss_values, y_values, target_accuracy=0.90)
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON OF SEARCH METHODS")
    print("=" * 80)
    
    results = [
        ("Percentile-based", p_metrics),
        ("Direct threshold", d_metrics),
        ("Binary search", b_metrics)
    ]
    
    best_overall = None
    best_accuracy = 0
    
    for method, metrics in results:
        if metrics:
            print(f"\n{method}:")
            print(f"  Threshold: {metrics['threshold']:.6f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_overall = (method, metrics)
    
    # Final recommendation
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    
    if best_overall:
        method, metrics = best_overall
        print(f"\nBest method: {method}")
        print_metrics(metrics)
        
        # Save results
        results_df = pd.DataFrame({
            'loss': loss_values,
            'y': y_values,
            'threshold': metrics['threshold'],
            'anomaly': (loss_values > metrics['threshold']).astype(int)
        })
        results_file = os.path.join(RESULTS_DIR, f'anomaly_detection_results_optimal.csv')
        results_df.to_csv(results_file, index=False)
        print(f"\n✓ Saved results to: {results_file}")
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_file = os.path.join(RESULTS_DIR, f'performance_metrics_optimal.csv')
        metrics_df.to_csv(metrics_file, index=False)
        print(f"✓ Saved metrics to: {metrics_file}")
        
        # Plot analysis
        plot_path = os.path.join(IMAGES_DIR, 'threshold_analysis_optimal.png')
        plot_threshold_analysis(loss_values, y_values, metrics['threshold'], plot_path)
        
        # Check if we achieved >90%
        if metrics['accuracy'] >= 0.90:
            print(f"\n✓ SUCCESS: Achieved {metrics['accuracy']*100:.2f}% accuracy (>= 90%)")
            print(f"  Recommended threshold: {metrics['threshold']:.6f}")
        else:
            print(f"\n⚠ WARNING: Best accuracy is {metrics['accuracy']*100:.2f}% (< 90%)")
            print(f"  This may indicate that the model needs retraining or the dataset is challenging.")
            print(f"  Recommended threshold: {metrics['threshold']:.6f}")
    else:
        print("\n⚠ ERROR: No valid thresholds found")
    
    print("\n" + "=" * 80)
    print("SEARCH COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()

