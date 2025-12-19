import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sns.set_style('whitegrid')

def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics over epochs
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_f1'
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[2].plot(epochs, history['val_f1'], 'g-o', label='Val F1', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_active_learning_progress(iteration_history, save_path=None):
    """
    Plot active learning progress over iterations
    
    Args:
        iteration_history: List of dicts with iteration metrics
        save_path: Path to save figure
    """
    df = pd.DataFrame(iteration_history)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training set size vs F1
    axes[0].plot(df['iteration'], df['train_size'], 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Training Set Size', fontsize=12)
    axes[0].set_title('Training Data Growth', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # F1 score progression
    if 'val_f1' in df.columns:
        axes[1].plot(df['train_size'], df['val_f1'], 'g-o', linewidth=2, markersize=8)
        axes[1].set_xlabel('Training Set Size', fontsize=12)
        axes[1].set_ylabel('Validation F1 Score', fontsize=12)
        axes[1].set_title('Performance vs Training Data', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_embeddings_tsne(embeddings, labels, n_classes=10, 
                         perplexity=30, save_path=None):
    """
    Visualize embeddings using t-SNE
    
    Args:
        embeddings: Embedding vectors [n_samples, embedding_dim]
        labels: Labels for each sample
        n_classes: Number of classes to show (if more, sample randomly)
        perplexity: t-SNE perplexity parameter
        save_path: Path to save figure
    """
    # Sample classes if too many
    unique_labels = np.unique(labels)
    if len(unique_labels) > n_classes:
        selected_classes = np.random.choice(unique_labels, n_classes, replace=False)
        mask = np.isin(labels, selected_classes)
        embeddings = embeddings[mask]
        labels = labels[mask]
    
    print("Running t-SNE... (this may take a while)")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Visualization of Embeddings', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_embeddings_pca(embeddings, labels, n_classes=10, save_path=None):
    """
    Visualize embeddings using PCA
    
    Args:
        embeddings: Embedding vectors [n_samples, embedding_dim]
        labels: Labels for each sample
        n_classes: Number of classes to show
        save_path: Path to save figure
    """
    # Sample classes if too many
    unique_labels = np.unique(labels)
    if len(unique_labels) > n_classes:
        selected_classes = np.random.choice(unique_labels, n_classes, replace=False)
        mask = np.isin(labels, selected_classes)
        embeddings = embeddings[mask]
        labels = labels[mask]
    
    print("Running PCA...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Class')
    plt.title(f'PCA Visualization of Embeddings\nExplained Variance: {pca.explained_variance_ratio_.sum():.2%}', 
             fontsize=16, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_class_distribution(labels, class_names=None, top_k=20, save_path=None):
    """
    Plot distribution of classes
    
    Args:
        labels: Array of labels
        class_names: Optional class names
        top_k: Show top K most frequent classes
        save_path: Path to save figure
    """
    # Count classes
    unique, counts = np.unique(labels, return_counts=True)
    
    # Sort by frequency
    sorted_idx = np.argsort(counts)[-top_k:]
    top_classes = unique[sorted_idx]
    top_counts = counts[sorted_idx]
    
    if class_names is not None:
        top_names = [class_names[i] for i in top_classes]
    else:
        top_names = [f"Class {i}" for i in top_classes]
    
    # Plot
    plt.figure(figsize=(12, max(6, top_k * 0.3)))
    bars = plt.barh(range(len(top_counts)), top_counts, color='steelblue')
    plt.yticks(range(len(top_counts)), top_names)
    plt.xlabel('Number of Samples', fontsize=12)
    plt.title(f'Top {top_k} Class Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, top_counts)):
        plt.text(count, i, f' {count}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_uncertainty_distribution(uncertainties, n_selected, save_path=None):
    """
    Plot distribution of uncertainty scores
    
    Args:
        uncertainties: Array of uncertainty values
        n_selected: Number of samples selected (to show threshold)
        save_path: Path to save figure
    """
    threshold = np.sort(uncertainties)[-n_selected]
    
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.hist(uncertainties, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Threshold line
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Selection Threshold (top {n_selected})')
    
    plt.xlabel('Uncertainty Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Uncertainty Score Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f'Mean: {uncertainties.mean():.3f}\nStd: {uncertainties.std():.3f}\nMedian: {np.median(uncertainties):.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example training history
    history = {
        'train_loss': [0.8, 0.6, 0.5, 0.4, 0.35],
        'val_loss': [0.85, 0.65, 0.55, 0.50, 0.48],
        'train_acc': [0.70, 0.78, 0.82, 0.85, 0.87],
        'val_acc': [0.68, 0.75, 0.79, 0.82, 0.83],
        'val_f1': [0.65, 0.73, 0.77, 0.80, 0.81]
    }
    
    # plot_training_history(history)
    
    # Example embeddings
    # embeddings = np.random.randn(1000, 128)
    # labels = np.random.randint(0, 10, 1000)
    # plot_embeddings_pca(embeddings, labels)
    
    pass