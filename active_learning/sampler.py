import torch
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

class ActiveLearningSampler:
    """
    Active learning sampler using uncertainty-based strategies
    """
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: Trained model for uncertainty estimation
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def get_predictions(self, dataloader, return_probs=True):
        """
        Get model predictions for a dataset
        
        Args:
            dataloader: DataLoader for unlabeled data
            return_probs: If True, return probabilities; else return logits
            
        Returns:
            predictions: numpy array of predictions
            indices: original indices from dataloader
        """
        all_probs = []
        all_indices = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc='Getting predictions')):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Handle multimodal
                if 'image' in batch:
                    images = batch['image'].to(self.device)
                    logits = self.model(input_ids, attention_mask, images)
                else:
                    logits = self.model(input_ids, attention_mask)
                
                # Convert to probabilities
                if return_probs:
                    probs = torch.softmax(logits, dim=1)
                else:
                    probs = logits
                
                all_probs.append(probs.cpu().numpy())
                
                # Track original indices
                batch_size = input_ids.size(0)
                batch_indices = range(batch_idx * dataloader.batch_size, 
                                    batch_idx * dataloader.batch_size + batch_size)
                all_indices.extend(batch_indices)
        
        predictions = np.vstack(all_probs)
        return predictions, np.array(all_indices)
    
    def entropy_sampling(self, probs):
        """
        Calculate entropy for each prediction
        Higher entropy = more uncertainty
        
        Args:
            probs: Probability predictions [n_samples, n_classes]
            
        Returns:
            uncertainties: Entropy values for each sample
        """
        # Calculate entropy for each sample
        uncertainties = entropy(probs.T)  # scipy entropy expects classes as first dim
        return uncertainties
    
    def margin_sampling(self, probs):
        """
        Calculate margin between top 2 predictions
        Lower margin = more uncertainty
        
        Args:
            probs: Probability predictions [n_samples, n_classes]
            
        Returns:
            uncertainties: Negative margin (higher = more uncertain)
        """
        # Sort probabilities
        sorted_probs = np.sort(probs, axis=1)
        
        # Margin between top 2
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        # Return negative (so higher values = more uncertain)
        uncertainties = -margins
        return uncertainties
    
    def least_confidence_sampling(self, probs):
        """
        Use 1 - max(prob) as uncertainty
        
        Args:
            probs: Probability predictions [n_samples, n_classes]
            
        Returns:
            uncertainties: 1 - confidence
        """
        confidences = np.max(probs, axis=1)
        uncertainties = 1 - confidences
        return uncertainties
    
    def select_samples(self, dataloader, n_samples, strategy='entropy'):
        """
        Select most uncertain samples for labeling
        
        Args:
            dataloader: DataLoader for unlabeled pool
            n_samples: Number of samples to select
            strategy: 'entropy', 'margin', or 'least_confidence'
            
        Returns:
            selected_indices: Indices of selected samples
            uncertainties: Uncertainty scores
        """
        print(f"\nSelecting {n_samples} samples using {strategy} sampling...")
        
        # Get predictions
        probs, indices = self.get_predictions(dataloader, return_probs=True)
        
        # Calculate uncertainties based on strategy
        if strategy == 'entropy':
            uncertainties = self.entropy_sampling(probs)
        elif strategy == 'margin':
            uncertainties = self.margin_sampling(probs)
        elif strategy == 'least_confidence':
            uncertainties = self.least_confidence_sampling(probs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Select top uncertain samples
        top_uncertain_idx = np.argsort(uncertainties)[-n_samples:]
        selected_indices = indices[top_uncertain_idx]
        selected_uncertainties = uncertainties[top_uncertain_idx]
        
        print(f"Selected indices: {selected_indices[:10]}... (showing first 10)")
        print(f"Uncertainty range: [{uncertainties.min():.4f}, {uncertainties.max():.4f}]")
        print(f"Selected uncertainty range: [{selected_uncertainties.min():.4f}, {selected_uncertainties.max():.4f}]")
        
        return selected_indices, uncertainties
    
    def random_sampling(self, dataloader, n_samples):
        """
        Random sampling baseline (no uncertainty)
        
        Args:
            dataloader: DataLoader for unlabeled pool
            n_samples: Number of samples to select
            
        Returns:
            selected_indices: Randomly selected indices
        """
        total_samples = len(dataloader.dataset)
        selected_indices = np.random.choice(total_samples, n_samples, replace=False)
        
        print(f"Randomly selected {n_samples} samples")
        return selected_indices


class ActiveLearningLoop:
    """
    Complete active learning workflow
    """
    def __init__(self, model, train_df, unlabeled_df, 
                 sampler, device='cuda'):
        """
        Args:
            model: Initial model
            train_df: Initially labeled data
            unlabeled_df: Pool of unlabeled data
            sampler: ActiveLearningSampler instance
            device: Device for training
        """
        self.model = model
        self.train_df = train_df.copy()
        self.unlabeled_df = unlabeled_df.copy()
        self.sampler = sampler
        self.device = device
        
        self.iteration_history = []
    
    def run_iteration(self, n_samples, strategy='entropy', retrain=True):
        """
        Run one active learning iteration
        
        Args:
            n_samples: Number of samples to label
            strategy: Sampling strategy
            retrain: Whether to retrain model after adding samples
            
        Returns:
            selected_samples: DataFrame of selected samples
        """
        print(f"\n{'='*60}")
        print(f"Active Learning Iteration {len(self.iteration_history) + 1}")
        print(f"{'='*60}")
        print(f"Current training size: {len(self.train_df)}")
        print(f"Unlabeled pool size: {len(self.unlabeled_df)}")
        
        # Create dataloader for unlabeled data
        # (You would create this using your dataset classes)
        # unlabeled_loader = create_unlabeled_dataloader(self.unlabeled_df)
        
        # Select samples
        # selected_indices, uncertainties = self.sampler.select_samples(
        #     unlabeled_loader, n_samples, strategy
        # )
        
        # Move samples from unlabeled to training
        # selected_samples = self.unlabeled_df.iloc[selected_indices].copy()
        # self.train_df = pd.concat([self.train_df, selected_samples], ignore_index=True)
        # self.unlabeled_df = self.unlabeled_df.drop(self.unlabeled_df.index[selected_indices]).reset_index(drop=True)
        
        # Record iteration
        # self.iteration_history.append({
        #     'iteration': len(self.iteration_history) + 1,
        #     'n_samples_added': n_samples,
        #     'train_size': len(self.train_df),
        #     'unlabeled_size': len(self.unlabeled_df),
        #     'strategy': strategy
        # })
        
        # Retrain if requested
        # if retrain:
        #     print("\nRetraining model with augmented dataset...")
        #     # Retrain model here
        
        # return selected_samples
        pass
    
    def run_full_loop(self, n_iterations, samples_per_iteration, strategy='entropy'):
        """
        Run complete active learning loop
        
        Args:
            n_iterations: Number of AL iterations
            samples_per_iteration: Samples to add each iteration
            strategy: Sampling strategy
        """
        for i in range(n_iterations):
            self.run_iteration(samples_per_iteration, strategy, retrain=True)
            
            # Evaluate after each iteration
            # val_metrics = evaluate_model(self.model, val_loader)
            # self.iteration_history[-1]['val_f1'] = val_metrics['f1']
            
        print("\nActive learning completed!")
        print(f"Final training size: {len(self.train_df)}")
        return self.iteration_history


# Example usage
if __name__ == "__main__":
    # Setup
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = MultimodalClassifier(num_classes=100).to(device)
    
    # Initialize sampler
    # sampler = ActiveLearningSampler(model, device)
    
    # Select samples
    # selected_idx, uncertainties = sampler.select_samples(
    #     unlabeled_loader,
    #     n_samples=1000,
    #     strategy='entropy'
    # )
    
    # Or run full loop
    # al_loop = ActiveLearningLoop(model, train_df, unlabeled_df, sampler)
    # history = al_loop.run_full_loop(
    #     n_iterations=5,
    #     samples_per_iteration=500,
    #     strategy='entropy'
    # )
    
    pass