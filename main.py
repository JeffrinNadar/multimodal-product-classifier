"""
Main script for Multimodal Product Classification with Active Learning

This script demonstrates the complete pipeline:
1. Data preprocessing
2. Baseline ML models
3. Deep learning models (text, image, multimodal)
4. Active learning
5. Evaluation and visualization
"""

import torch
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

# Import custom modules
from data_pipeline.text_preprocess import TextPreprocessor
from data_pipeline.image_preprocess import ImagePreprocessor
from data_pipeline.dataset import create_dataloaders
from models.baseline_ml import BaselineModels
from models.text_encoder import TextClassifier
from models.image_encoder import ImageClassifier
from models.multimodal_classifier import MultimodalClassifier
from training.train_pytorch import Trainer, evaluate_model
from active_learning.sampler import ActiveLearningSampler, ActiveLearningLoop
from utils.metrics import calculate_metrics, print_metrics, plot_confusion_matrix
from utils.visualization import plot_training_history, plot_active_learning_progress

def setup_directories():
    """Create necessary directories"""
    dirs = ['data', 'models_saved', 'logs', 'results', 'figures']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multimodal Product Classification')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--text_column', type=str, default='title', help='Text column name')
    parser.add_argument('--label_column', type=str, default='category', help='Label column name')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, choices=['text', 'image', 'multimodal'], 
                       default='multimodal', help='Type of model to train')
    parser.add_argument('--text_model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--image_model', type=str, default='resnet50')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--patience', type=int, default=3)
    
    # Active learning arguments
    parser.add_argument('--use_active_learning', action='store_true')
    parser.add_argument('--al_iterations', type=int, default=5)
    parser.add_argument('--al_samples_per_iter', type=int, default=500)
    parser.add_argument('--al_strategy', type=str, default='entropy', 
                       choices=['entropy', 'margin', 'least_confidence'])
    
    # Other
    parser.add_argument('--run_baseline', action='store_true', help='Run baseline ML models')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    """Main execution pipeline"""
    
    # Parse arguments
    args = parse_args()
    
    # Setup
    setup_directories()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ========== STEP 1: Data Preprocessing ==========
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Text preprocessing
    text_prep = TextPreprocessor(max_length=128)
    df_processed = text_prep.process_dataframe(
        df,
        text_column=args.text_column,
        label_column=args.label_column
    )
    
    # Create splits
    train_df, val_df, test_df = text_prep.create_splits(df_processed)
    
    # Image preprocessing (if using images)
    image_prep = None
    if args.model_type in ['image', 'multimodal'] and args.image_dir:
        print("\nProcessing images...")
        image_prep = ImagePreprocessor(image_size=224)
        train_df = image_prep.create_image_paths_column(train_df, args.image_dir)
        val_df = image_prep.create_image_paths_column(val_df, args.image_dir)
        test_df = image_prep.create_image_paths_column(test_df, args.image_dir)
        
        # Verify images
        train_df = image_prep.verify_images(train_df, args.image_dir)
        val_df = image_prep.verify_images(val_df, args.image_dir)
        test_df = image_prep.verify_images(test_df, args.image_dir)
    
    num_classes = text_prep.num_classes
    print(f"\nNumber of classes: {num_classes}")
    
    # ========== STEP 2: Baseline Models (Optional) ==========
    if args.run_baseline:
        print("\n" + "="*70)
        print("STEP 2: BASELINE ML MODELS")
        print("="*70)
        
        baseline = BaselineModels(max_features=10000)
        X_train, X_val, X_test, y_train, y_val, y_test = baseline.prepare_data(
            train_df, val_df, test_df
        )
        
        baseline.train_all(X_train, y_train, X_val, y_val)
        baseline.evaluate_test(X_test, y_test)
        
        # Save results
        results_df = baseline.get_results_df()
        results_df.to_csv('results/baseline_results.csv', index=False)
        print("\nBaseline results saved to results/baseline_results.csv")
    
    # ========== STEP 3: Deep Learning Model ==========
    print("\n" + "="*70)
    print(f"STEP 3: DEEP LEARNING MODEL ({args.model_type.upper()})")
    print("="*70)
    
    # Create dataloaders
    use_multimodal = args.model_type == 'multimodal'
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        text_prep, image_prep,
        batch_size=args.batch_size,
        num_workers=4,
        multimodal=use_multimodal
    )
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    if args.model_type == 'text':
        model = TextClassifier(
            num_classes=num_classes,
            model_name=args.text_model
        )
    elif args.model_type == 'image':
        model = ImageClassifier(
            num_classes=num_classes,
            model_name=args.image_model
        )
    else:  # multimodal
        model = MultimodalClassifier(
            num_classes=num_classes,
            text_model=args.text_model,
            image_model=args.image_model
        )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    trainer = Trainer(
        model, train_loader, val_loader,
        device=device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        log_dir=f'logs/{args.model_type}_experiment'
    )
    
    history = trainer.train()
    
    # Plot training history
    plot_training_history(history, save_path=f'figures/{args.model_type}_training_history.png')
    
    # ========== STEP 4: Evaluation ==========
    print("\n" + "="*70)
    print("STEP 4: EVALUATION")
    print("="*70)
    
    # Load best model
    trainer.load_checkpoint('best_model.pt')
    
    # Evaluate on test set
    test_results = evaluate_model(model, test_loader, device=device)
    
    # Calculate detailed metrics
    metrics = calculate_metrics(
        test_results['labels'],
        test_results['predictions']
    )
    print_metrics(metrics, f"{args.model_type.upper()} Model - Test Set")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_results['labels'],
        test_results['predictions'],
        normalize=True,
        save_path=f'figures/{args.model_type}_confusion_matrix.png'
    )
    
    # Save model
    torch.save(model.state_dict(), f'models_saved/{args.model_type}_final.pt')
    print(f"\nModel saved to models_saved/{args.model_type}_final.pt")
    
    # ========== STEP 5: Active Learning (Optional) ==========
    if args.use_active_learning:
        print("\n" + "="*70)
        print("STEP 5: ACTIVE LEARNING")
        print("="*70)
        
        # Initialize sampler
        sampler = ActiveLearningSampler(model, device=device)
        
        # Split train into labeled and unlabeled
        initial_labeled_size = len(train_df) // 10  # Start with 10% labeled
        initial_indices = np.random.choice(len(train_df), initial_labeled_size, replace=False)
        
        labeled_df = train_df.iloc[initial_indices].reset_index(drop=True)
        unlabeled_df = train_df.drop(train_df.index[initial_indices]).reset_index(drop=True)
        
        print(f"Initial labeled: {len(labeled_df)}")
        print(f"Unlabeled pool: {len(unlabeled_df)}")
        
        # Run active learning loop
        al_loop = ActiveLearningLoop(model, labeled_df, unlabeled_df, sampler, device)
        al_history = al_loop.run_full_loop(
            n_iterations=args.al_iterations,
            samples_per_iteration=args.al_samples_per_iter,
            strategy=args.al_strategy
        )
        
        # Plot active learning progress
        plot_active_learning_progress(
            al_history,
            save_path='figures/active_learning_progress.png'
        )
        
        # Save active learning results
        pd.DataFrame(al_history).to_csv('results/active_learning_history.csv', index=False)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nResults saved to:")
    print("  - models_saved/")
    print("  - results/")
    print("  - figures/")
    print("  - logs/ (TensorBoard)")
    print("\nTo view TensorBoard logs, run:")
    print(f"  tensorboard --logdir=logs/{args.model_type}_experiment")

if __name__ == "__main__":
    main()