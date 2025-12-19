"""
Download and prepare Amazon Product Reviews dataset.
This script downloads the dataset from HuggingFace and saves it locally.
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm


def download_amazon_reviews(output_dir: str, category: str = "all", sample_size: int = None):
    """
    Download Amazon US Reviews dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save the dataset
        category: Product category to download (e.g., 'Electronics', 'Books', 'all')
        sample_size: Number of samples to download (None = all)
    """
    print(f"📦 Downloading Amazon Reviews dataset...")
    print(f"   Category: {category}")
    print(f"   Sample size: {sample_size if sample_size else 'All'}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Available categories in amazon_us_reviews
    available_categories = [
        "Wireless", "Watches", "Video_Games", "Video_DVD", "Video",
        "Toys", "Tools", "Sports", "Software", "Shoes", "Pet_Products",
        "Personal_Care_Appliances", "PC", "Outdoors", "Office_Products",
        "Musical_Instruments", "Music", "Mobile_Electronics", "Mobile_Apps",
        "Major_Appliances", "Luggage", "Lawn_and_Garden", "Kitchen",
        "Jewelry", "Home_Improvement", "Home_Entertainment", "Home",
        "Health_Personal_Care", "Grocery", "Gift_Card", "Furniture",
        "Electronics", "Digital_Video_Games", "Digital_Video_Download",
        "Digital_Software", "Digital_Music_Purchase", "Digital_Ebook_Purchase",
        "Camera", "Books", "Beauty", "Baby", "Automotive", "Apparel"
    ]
    
    if category == "all":
        print(f"\n⚠️  Downloading ALL categories. This will take significant time and space.")
        print(f"   Available categories: {len(available_categories)}")
        categories_to_download = available_categories[:5]  # Start with first 5 for testing
        print(f"   Starting with: {categories_to_download}")
    else:
        categories_to_download = [category]
    
    all_data = []
    
    for cat in tqdm(categories_to_download, desc="Downloading categories"):
        try:
            print(f"\n📂 Loading {cat}...")
            
            # Load dataset from HuggingFace
            dataset = load_dataset(
                "amazon_us_reviews",
                cat,
                split="train",
                trust_remote_code=True
            )
            
            print(f"   ✅ Loaded {len(dataset)} samples from {cat}")
            
            # Sample if needed
            if sample_size:
                dataset = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))
                print(f"   📊 Sampled to {len(dataset)} samples")
            
            # Convert to pandas for easier manipulation
            df = dataset.to_pandas()
            df['category'] = cat
            all_data.append(df)
            
        except Exception as e:
            print(f"   ❌ Error loading {cat}: {e}")
            continue
    
    # Combine all categories
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ Total samples collected: {len(combined_df)}")
        
        # Save to disk
        output_file = output_path / "amazon_reviews_raw.parquet"
        combined_df.to_parquet(output_file, index=False)
        print(f"💾 Saved to: {output_file}")
        
        # Print dataset statistics
        print("\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(combined_df):,}")
        print(f"   Categories: {combined_df['category'].nunique()}")
        print(f"   Columns: {list(combined_df.columns)}")
        print(f"\n   Category distribution:")
        print(combined_df['category'].value_counts())
        
        # Save a sample for quick inspection
        sample_file = output_path / "amazon_reviews_sample.csv"
        combined_df.head(1000).to_csv(sample_file, index=False)
        print(f"\n💾 Sample saved to: {sample_file}")
        
        return combined_df
    else:
        print("❌ No data downloaded!")
        return None


def get_dataset_info():
    """Print information about available Amazon review datasets."""
    print("\n📚 Amazon US Reviews Dataset Information")
    print("=" * 60)
    print("\nAvailable Categories (40 total):")
    print("  • Electronics, Books, Clothing")
    print("  • Home & Kitchen, Sports, Toys")
    print("  • Video Games, Music, Beauty")
    print("  • And 31 more categories...")
    print("\nDataset Fields:")
    print("  • product_id: Unique product identifier")
    print("  • product_title: Name of the product")
    print("  • product_category: Category classification")
    print("  • star_rating: 1-5 star rating")
    print("  • review_headline: Summary of the review")
    print("  • review_body: Full review text")
    print("  • helpful_votes: Number of helpful votes")
    print("  • total_votes: Total number of votes")
    print("\nRecommended starter categories:")
    print("  • Electronics: ~3M reviews (good variety)")
    print("  • Books: ~3M reviews (text-heavy)")
    print("  • Video_Games: ~1.7M reviews (manageable size)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Amazon Product Reviews dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/raw",
        help="Directory to save downloaded data"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Electronics",
        help="Category to download (e.g., Electronics, Books, all)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100000,
        help="Number of samples to download per category (None for all)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show dataset information and exit"
    )
    
    args = parser.parse_args()
    
    if args.info:
        get_dataset_info()
    else:
        download_amazon_reviews(
            output_dir=args.output_dir,
            category=args.category,
            sample_size=args.sample_size
        )
        print("\n✨ Download complete! Next steps:")
        print("   1. Run exploratory data analysis")
        print("   2. Preprocess the text data")
        print("   3. Create train/val/test splits")