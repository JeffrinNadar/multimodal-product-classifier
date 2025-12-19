import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

class MultimodalDataset(Dataset):
    """
    Dataset for multimodal product classification (text + images)
    """
    def __init__(self, df, text_preprocessor, image_preprocessor, 
                 text_column='text', image_column='image_path', 
                 label_column='label', train=True):
        """
        Args:
            df: Dataframe with text, image paths, and labels
            text_preprocessor: TextPreprocessor instance
            image_preprocessor: ImagePreprocessor instance
            text_column: Column name for text data
            image_column: Column name for image paths
            label_column: Column name for labels
            train: Whether this is training data (affects augmentation)
        """
        self.df = df.reset_index(drop=True)
        self.text_preprocessor = text_preprocessor
        self.image_preprocessor = image_preprocessor
        self.text_column = text_column
        self.image_column = image_column
        self.label_column = label_column
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get text
        text = row[self.text_column]
        
        # Tokenize text
        tokens = self.text_preprocessor.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.text_preprocessor.max_length,
            return_tensors='pt'
        )
        
        # Get image
        image_path = row[self.image_column]
        image = self.image_preprocessor.process_image(image_path, train=self.train)
        
        # Handle missing images
        if image is None:
            image = torch.zeros(3, self.image_preprocessor.image_size, 
                               self.image_preprocessor.image_size)
        
        # Get label
        label = torch.tensor(row[self.label_column], dtype=torch.long)
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'image': image,
            'label': label
        }


class TextOnlyDataset(Dataset):
    """
    Dataset for text-only classification (for baseline models)
    """
    def __init__(self, df, text_preprocessor, 
                 text_column='text', label_column='label'):
        """
        Args:
            df: Dataframe with text and labels
            text_preprocessor: TextPreprocessor instance
            text_column: Column name for text data
            label_column: Column name for labels
        """
        self.df = df.reset_index(drop=True)
        self.text_preprocessor = text_preprocessor
        self.text_column = text_column
        self.label_column = label_column
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get text
        text = row[self.text_column]
        
        # Tokenize text
        tokens = self.text_preprocessor.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.text_preprocessor.max_length,
            return_tensors='pt'
        )
        
        # Get label
        label = torch.tensor(row[self.label_column], dtype=torch.long)
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'label': label
        }


def create_dataloaders(train_df, val_df, test_df, 
                       text_preprocessor, image_preprocessor,
                       batch_size=32, num_workers=4, multimodal=True):
    """
    Create PyTorch DataLoaders for train/val/test sets
    
    Args:
        train_df, val_df, test_df: Dataframes for each split
        text_preprocessor: TextPreprocessor instance
        image_preprocessor: ImagePreprocessor instance
        batch_size: Batch size for dataloaders
        num_workers: Number of parallel workers
        multimodal: If True, use MultimodalDataset; else TextOnlyDataset
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    if multimodal:
        train_dataset = MultimodalDataset(
            train_df, text_preprocessor, image_preprocessor, train=True
        )
        val_dataset = MultimodalDataset(
            val_df, text_preprocessor, image_preprocessor, train=False
        )
        test_dataset = MultimodalDataset(
            test_df, text_preprocessor, image_preprocessor, train=False
        )
    else:
        train_dataset = TextOnlyDataset(train_df, text_preprocessor)
        val_dataset = TextOnlyDataset(val_df, text_preprocessor)
        test_dataset = TextOnlyDataset(test_df, text_preprocessor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    from preprocess_text import TextPreprocessor
    from preprocess_image import ImagePreprocessor
    
    # Initialize preprocessors
    text_prep = TextPreprocessor()
    image_prep = ImagePreprocessor()
    
    # Load data
    # train_df = pd.read_csv('data/train.csv')
    # val_df = pd.read_csv('data/val.csv')
    # test_df = pd.read_csv('data/test.csv')
    
    # Create dataloaders
    # train_loader, val_loader, test_loader = create_dataloaders(
    #     train_df, val_df, test_df,
    #     text_prep, image_prep,
    #     batch_size=32,
    #     num_workers=4
    # )
    
    # Test iteration
    # for batch in train_loader:
    #     print(f"Input IDs shape: {batch['input_ids'].shape}")
    #     print(f"Image shape: {batch['image'].shape}")
    #     print(f"Label shape: {batch['label'].shape}")
    #     break
    
    pass