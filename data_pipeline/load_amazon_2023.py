"""
Data loader for Amazon Reviews 2023 dataset from HuggingFace
Handles both reviews and metadata with images
"""

from datasets import load_dataset
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
from tqdm import tqdm
import json

class AmazonReviews2023Loader:
    """
    Loader for Amazon Reviews 2023 dataset
    """
    def __init__(self, category='All_Beauty', download_images=True, 
                 image_dir='data/amazon_images', sample_size=None):
        """
        Args:
            category: Category name (e.g., 'All_Beauty', 'Electronics', etc.)
            download_images: Whether to download product images
            image_dir: Directory to save images
            sample_size: If set, sample this many items (for testing)
        """
        self.category = category
        self.download_images = download_images
        self.image_dir = image_dir
        self.sample_size = sample_size
        
        if download_images:
            os.makedirs(image_dir, exist_ok=True)
    
    def load_reviews(self):
        """
        Load user reviews from HuggingFace
        
        Returns:
            DataFrame with reviews
        """
        print(f"Loading reviews for category: {self.category}")
        
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", 
            f"raw_review_{self.category}",
            split="full",
            trust_remote_code=True
        )
        
        # Convert to pandas
        df = pd.DataFrame(dataset)
        
        print(f"Loaded {len(df)} reviews")
        
        # Sample if needed
        if self.sample_size and len(df) > self.sample_size:
            print(f"Sampling {self.sample_size} reviews...")
            df = df.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
        
        return df
    
    def load_metadata(self):
        """
        Load item metadata from HuggingFace
        
        Returns:
            DataFrame with product metadata
        """
        print(f"Loading metadata for category: {self.category}")
        
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{self.category}",
            split="full",
            trust_remote_code=True
        )
        
        # Convert to pandas
        df = pd.DataFrame(dataset)
        
        print(f"Loaded {len(df)} products")
        
        return df
    
    def download_product_image(self, image_url, parent_asin):
        """
        Download a single product image
        
        Args:
            image_url: URL of the image
            parent_asin: Product ID
            
        Returns:
            Path to saved image or None if failed
        """
        if not image_url or image_url == 'None':
            return None
        
        try:
            response = requests.get(image_url, timeout=5)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save image
                image_path = os.path.join(self.image_dir, f"{parent_asin}.jpg")
                img.save(image_path)
                return image_path
        except Exception as e:
            print(f"Error downloading {image_url}: {e}")
        
        return None
    
    def download_images_for_metadata(self, meta_df):
        """
        Download images for all products in metadata
        
        Args:
            meta_df: Metadata dataframe
            
        Returns:
            DataFrame with image_path column added
        """
        print("Downloading product images...")
        
        image_paths = []
        
        for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
            parent_asin = row['parent_asin']
            
            # Check if image already exists
            existing_path = os.path.join(self.image_dir, f"{parent_asin}.jpg")
            if os.path.exists(existing_path):
                image_paths.append(existing_path)
                continue
            
            # Try to download from available images
            image_path = None
            if 'images' in row and row['images']:
                images_dict = row['images']
                
                # Try large images first
                if 'large' in images_dict and images_dict['large']:
                    for url in images_dict['large']:
                        if url and url != 'None':
                            image_path = self.download_product_image(url, parent_asin)
                            if image_path:
                                break
                
                # Try hi_res if large failed
                if not image_path and 'hi_res' in images_dict and images_dict['hi_res']:
                    for url in images_dict['hi_res']:
                        if url and url != 'None':
                            image_path = self.download_product_image(url, parent_asin)
                            if image_path:
                                break
            
            image_paths.append(image_path)
        
        meta_df['image_path'] = image_paths
        
        # Count successful downloads
        valid_images = sum(1 for p in image_paths if p is not None)
        print(f"Successfully downloaded {valid_images}/{len(meta_df)} images")
        
        return meta_df
    
    def create_training_dataset(self, max_reviews_per_product=5):
        """
        Create training dataset by combining reviews and metadata
        
        Args:
            max_reviews_per_product: Max reviews to keep per product
            
        Returns:
            Combined DataFrame ready for training
        """
        # Load both datasets
        reviews_df = self.load_reviews()
        meta_df = self.load_metadata()
        
        # Download images if requested
        if self.download_images:
            meta_df = self.download_images_for_metadata(meta_df)
        
        print("\nMerging reviews with metadata...")
        
        # Aggregate reviews by product
        # Take the most recent reviews per product
        reviews_df = reviews_df.sort_values('timestamp', ascending=False)
        reviews_grouped = reviews_df.groupby('parent_asin').head(max_reviews_per_product)
        
        # Combine title and text for each review
        reviews_grouped['review_text'] = (
            reviews_grouped['title'].fillna('') + ' ' + reviews_grouped['text'].fillna('')
        ).str.strip()
        
        # Merge with metadata
        df = reviews_grouped.merge(
            meta_df[['parent_asin', 'title', 'description', 'features', 
                    'main_category', 'image_path']],
            on='parent_asin',
            how='inner',
            suffixes=('_review', '_product')
        )
        
        # Create combined text (product title + description + review)
        df['combined_text'] = (
            df['title_product'].fillna('') + ' ' +
            df['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '').fillna('') + ' ' +
            df['review_text'].fillna('')
        ).str.strip()
        
        # Rename for compatibility with existing code
        df = df.rename(columns={
            'combined_text': 'text',
            'main_category': 'category',
            'image_path': 'image_path'
        })
        
        # Keep only necessary columns
        df = df[['text', 'category', 'parent_asin', 'image_path', 'rating', 
                'verified_purchase', 'helpful_vote']].copy()
        
        # Remove items without images (if using multimodal)
        if self.download_images:
            before = len(df)
            df = df[df['image_path'].notna()].copy()
            after = len(df)
            print(f"Filtered to items with images: {after}/{before}")
        
        print(f"\nFinal dataset size: {len(df)}")
        print(f"Unique categories: {df['category'].nunique()}")
        print(f"Category distribution:")
        print(df['category'].value_counts())
        
        return df
    
    def save_dataset(self, df, output_dir='data'):
        """
        Save processed dataset
        
        Args:
            df: Processed dataframe
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f'amazon_{self.category}_processed.csv')
        df.to_csv(output_path, index=False)
        
        print(f"\nDataset saved to: {output_path}")
        
        # Save category mapping
        categories = sorted(df['category'].unique())
        category_map = {cat: idx for idx, cat in enumerate(categories)}
        
        map_path = os.path.join(output_dir, f'amazon_{self.category}_categories.json')
        with open(map_path, 'w') as f:
            json.dump(category_map, f, indent=2)
        
        print(f"Category mapping saved to: {map_path}")
        
        return output_path


# Example usage and quick start script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='All_Beauty',
                       help='Amazon category (All_Beauty, Electronics, etc.)')
    parser.add_argument('--sample_size', type=int, default=10000,
                       help='Sample size (None for full dataset)')
    parser.add_argument('--download_images', action='store_true',
                       help='Download product images')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create loader
    loader = AmazonReviews2023Loader(
        category=args.category,
        download_images=args.download_images,
        sample_size=args.sample_size
    )
    
    # Create dataset
    df = loader.create_training_dataset()
    
    # Save
    loader.save_dataset(df, output_dir=args.output_dir)
    
    print("\nDataset ready for training!")
    print(f"Use this file in main.py: {args.output_dir}/amazon_{args.category}_processed.csv")