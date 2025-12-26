import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm

class TextPreprocessor:
    def __init__(self, model_name='distilbert-base-uncased', max_length=128):
        """
        Initialize text preprocessor with tokenizer
        
        Args:
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def clean_text(self, text):
        """Clean and normalize text data"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process_dataframe(self, df, text_column='title', 
                         description_column=None, label_column='category'):
        """
        Process entire dataframe of text data
        
        Args:
            df: Input dataframe
            text_column: Name of main text column
            description_column: Optional description column to concatenate
            label_column: Name of label/category column
            
        Returns:
            Processed dataframe with cleaned text
        """
        print("Cleaning text data...")
        
        # Create combined text field
        df['text'] = df[text_column].apply(self.clean_text)
        
        if description_column and description_column in df.columns:
            df['description_clean'] = df[description_column].apply(self.clean_text)
            df['text'] = df['text'] + ' ' + df['description_clean']
        
        # Remove empty texts
        df = df[df['text'].str.len() > 0].copy()
        
        # Encode labels
        if label_column in df.columns:
            df['label'] = pd.Categorical(df[label_column]).codes
            self.label_map = dict(enumerate(df[label_column].astype('category').cat.categories))
            self.num_classes = len(self.label_map)
        
        return df
    
    def tokenize_batch(self, texts):
        """
        Tokenize a batch of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with input_ids, attention_mask tensors
        """
        return self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def create_splits(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """
        Create stratified train/val/test splits
        
        Args:
            df: Processed dataframe
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
            random_state: Random seed
            
        Returns:
            train_df, val_df, test_df
        """
        print("Creating stratified splits...")
        
        # First split: train+val vs test
        # Added code to filter out rare categories
        category_counts = df['category'].value_counts()
        valid_categories = category_counts[category_counts >= 2].index

        df = df[df['category'].isin(valid_categories)].reset_index(drop=True)

        removed = set(category_counts.index) - set(valid_categories)
        print(f"Removed {len(removed)} rare categories: {removed}")
        #### End of added code
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['label'],
            random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            stratify=train_val['label'],
            random_state=random_state
        )
        
        print(f"Train size: {len(train)}")
        print(f"Val size: {len(val)}")
        print(f"Test size: {len(test)}")
        
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# Example usage
if __name__ == "__main__":
    # Load your data
    # df = pd.read_csv('amazon_products.csv')
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(max_length=128)
    
    # Process data
    # df_processed = preprocessor.process_dataframe(
    #     df, 
    #     text_column='title',
    #     description_column='description',
    #     label_column='category'
    # )
    
    # Create splits
    # train_df, val_df, test_df = preprocessor.create_splits(df_processed)
    
    # Save processed data
    # train_df.to_csv('data/train.csv', index=False)
    # val_df.to_csv('data/val.csv', index=False)
    # test_df.to_csv('data/test.csv', index=False)
    
    pass