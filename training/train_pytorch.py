import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report

class Trainer:
    """
    Trainer class for PyTorch models
    """
    def __init__(self, model, train_loader, val_loader, 
                 device='cuda', learning_rate=2e-5,
                 num_epochs=10, patience=3,
                 log_dir='runs/experiment'):
        """
        Args:
            model: PyTorch model to train
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device to train on
            learning_rate: Learning rate
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            log_dir: Directory for TensorBoard logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Handle images if present (multimodal)
            if 'image' in batch:
                images = batch['image'].to(self.device)
                logits = self.model(input_ids, attention_mask, images)
            else:
                logits = self.model(input_ids, attention_mask)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Handle images if present
                if 'image' in batch:
                    images = batch['image'].to(self.device)
                    logits = self.model(input_ids, attention_mask, images)
                else:
                    logits = self.model(input_ids, attention_mask)
                
                # Compute loss
                loss = self.criterion(logits, labels)
                
                # Track metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def train(self):
        """Full training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_f1)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('F1/val', val_f1, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pt')
                print(f"New best model! Val F1: {val_f1:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best Val F1: {self.best_val_f1:.4f} at epoch {self.best_epoch + 1}")
                break
        
        self.writer.close()
        print("\nTraining completed!")
        return self.history
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'history': self.history
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_f1 = checkpoint['best_val_f1']
        self.history = checkpoint['history']


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: Device to use
        
    Returns:
        Dictionary with metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Evaluating on test set...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            if 'image' in batch:
                images = batch['image'].to(device)
                logits = model(input_ids, attention_mask, images)
            else:
                logits = model(input_ids, attention_mask)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels
    }


# Example usage
if __name__ == "__main__":
    # This would be in your main script
    # from models.multimodal_classifier import MultimodalClassifier
    # from data_pipeline.dataset import create_dataloaders
    
    # Setup
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # num_classes = 100
    
    # Create model
    # model = MultimodalClassifier(num_classes=num_classes)
    
    # Create dataloaders
    # train_loader, val_loader, test_loader = create_dataloaders(...)
    
    # Train
    # trainer = Trainer(model, train_loader, val_loader, device=device)
    # history = trainer.train()
    
    # Evaluate
    # results = evaluate_model(model, test_loader, device=device)
    
    pass