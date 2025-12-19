import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true, y_pred, y_probs=None, average='weighted'):
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (optional, for AUC)
        average: Averaging strategy for multi-class metrics
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Add per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['precision_per_class'] = precision_per_class
    metrics['recall_per_class'] = recall_per_class
    metrics['f1_per_class'] = f1_per_class
    
    # AUC if probabilities provided
    if y_probs is not None:
        try:
            n_classes = y_probs.shape[1]
            if n_classes == 2:
                # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_probs[:, 1])
            else:
                # Multi-class (one-vs-rest)
                metrics['auc'] = roc_auc_score(
                    y_true, y_probs, 
                    multi_class='ovr', 
                    average=average
                )
        except:
            metrics['auc'] = None
    
    return metrics


def print_metrics(metrics, title="Model Metrics"):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the output
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    if 'auc' in metrics and metrics['auc'] is not None:
        print(f"  AUC:       {metrics['auc']:.4f}")
    
    print(f"\n{'='*60}")


def plot_confusion_matrix(y_true, y_pred, labels=None, 
                         normalize=True, figsize=(10, 8), 
                         save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        normalize: Whether to normalize
        figsize: Figure size
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    
    # For large number of classes, show heatmap without annotations
    n_classes = len(np.unique(y_true))
    if n_classes > 20:
        sns.heatmap(cm, cmap='Blues', cbar=True, 
                   xticklabels=False, yticklabels=False)
    else:
        # Generate labels if not provided
        if labels is None:
            labels = [f'Class {i}' for i in range(n_classes)]
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_per_class_metrics(metrics, class_names=None, 
                           top_k=20, save_path=None):
    """
    Plot per-class F1 scores
    
    Args:
        metrics: Dictionary containing per-class metrics
        class_names: Optional class names
        top_k: Show top K classes by F1
        save_path: Path to save figure
    """
    f1_scores = metrics['f1_per_class']
    
    # Sort by F1 score
    sorted_indices = np.argsort(f1_scores)[-top_k:]
    sorted_f1 = f1_scores[sorted_indices]
    
    if class_names is not None:
        sorted_names = [class_names[i] for i in sorted_indices]
    else:
        sorted_names = [f"Class {i}" for i in sorted_indices]
    
    # Plot
    plt.figure(figsize=(12, max(6, top_k * 0.3)))
    bars = plt.barh(range(len(sorted_f1)), sorted_f1)
    
    # Color bars based on performance
    colors = ['red' if x < 0.5 else 'orange' if x < 0.7 else 'green' 
              for x in sorted_f1]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.yticks(range(len(sorted_f1)), sorted_names)
    plt.xlabel('F1 Score')
    plt.title(f'Top {top_k} Classes by F1 Score')
    plt.xlim([0, 1])
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_models(results_dict, metric='f1', save_path=None):
    """
    Compare multiple models
    
    Args:
        results_dict: Dictionary mapping model names to their metrics
        metric: Metric to compare
        save_path: Path to save figure
    """
    models = list(results_dict.keys())
    scores = [results_dict[m][metric] for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores)
    
    # Color best model
    best_idx = np.argmax(scores)
    bars[best_idx].set_color('green')
    
    plt.ylabel(metric.upper())
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def generate_classification_report(y_true, y_pred, class_names=None, 
                                   save_path=None):
    """
    Generate and optionally save classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
        save_path: Path to save report
        
    Returns:
        Classification report string
    """
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        digits=4
    )
    
    print(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {save_path}")
    
    return report


# Example usage
if __name__ == "__main__":
    # Generate dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 5, 1000)
    y_pred = y_true.copy()
    # Add some noise
    noise_idx = np.random.choice(1000, 200, replace=False)
    y_pred[noise_idx] = np.random.randint(0, 5, 200)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics)
    
    # Plot confusion matrix
    # plot_confusion_matrix(y_true, y_pred, normalize=True)
    
    # Generate report
    # generate_classification_report(y_true, y_pred)