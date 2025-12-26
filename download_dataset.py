"""
Download Amazon Reviews 2023 dataset directly from source
Run this BEFORE running main.py
"""

import os
import sys
import gzip
import json
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
import ast

# Direct download URLs from UCSD
BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw"

CATEGORIES = {
    'All_Beauty': 'All_Beauty',
    'Electronics': 'Electronics',
    'Toys_and_Games': 'Toys_and_Games',
    'Sports_and_Outdoors': 'Sports_and_Outdoors',
    'Clothing_Shoes_and_Jewelry': 'Clothing_Shoes_and_Jewelry',
    'Home_and_Kitchen': 'Home_and_Kitchen',
    'Health_and_Household': 'Health_and_Household',
    'Books': 'Books',
    'Movies_and_TV': 'Movies_and_TV',
    'Video_Games': 'Video_Games',
    'Pet_Supplies': 'Pet_Supplies',
    'Automotive': 'Automotive',
    'Office_Products': 'Office_Products',
}

def download_file(url, output_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def read_jsonl_gz(file_path, sample_size=None):
    """Read gzipped JSONL file"""
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            try:
                data.append(json.loads(line))
            except:
                continue
    return data


def download_amazon_dataset(category='All_Beauty', sample_size=10000, output_dir='data'):
    """
    Download and prepare Amazon Reviews dataset from source
    
    Args:
        category: Amazon category name
        sample_size: Number of samples to use
        output_dir: Output directory
    """
    
    print("="*70)
    print("Amazon Reviews 2023 Dataset Downloader")
    print("="*70)
    print(f"Category: {category}")
    print(f"Sample size: {sample_size if sample_size else 'FULL DATASET'}")
    print("="*70)
    
    if category not in CATEGORIES:
        print(f"\n✗ Category '{category}' not found!")
        print("\nAvailable categories:")
        for cat in CATEGORIES.keys():
            print(f"  - {cat}")
        sys.exit(1)
    
    # Create directories
    Path(output_dir).mkdir(exist_ok=True)
    temp_dir = Path(output_dir) / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # File paths
    review_file = temp_dir / f"{category}_reviews.jsonl.gz"
    meta_file = temp_dir / f"{category}_meta.jsonl.gz"
    
    # URLs
    review_url = f"{BASE_URL}/review_categories/{category}.jsonl.gz"
    meta_url = f"{BASE_URL}/meta_categories/meta_{category}.jsonl.gz"
    
    # Step 1: Download reviews
    print(f"\n[1/4] Downloading reviews...")
    if not review_file.exists():
        try:
            print(f"Downloading from: {review_url}")
            download_file(review_url, review_file)
            print("✓ Reviews downloaded")
        except Exception as e:
            print(f"✗ Error downloading reviews: {e}")
            print("\nTrying alternative approach...")
            print("Please manually download the dataset:")
            print(f"1. Go to: https://amazon-reviews-2023.github.io/")
            print(f"2. Download {category} reviews and metadata")
            print(f"3. Place files in: {temp_dir}/")
            sys.exit(1)
    else:
        print("✓ Reviews file already exists")
    
    # Step 2: Download metadata
    print(f"\n[2/4] Downloading metadata...")
    if not meta_file.exists():
        try:
            print(f"Downloading from: {meta_url}")
            download_file(meta_url, meta_file)
            print("✓ Metadata downloaded")
        except Exception as e:
            print(f"✗ Error downloading metadata: {e}")
            sys.exit(1)
    else:
        print("✓ Metadata file already exists")
    
    # Step 3: Parse reviews
    print(f"\n[3/4] Parsing reviews (this may take a minute)...")
    reviews_data = read_jsonl_gz(review_file, sample_size=sample_size)
    reviews_df = pd.DataFrame(reviews_data)
    print(f"✓ Loaded {len(reviews_df):,} reviews")
    
    # Step 4: Parse metadata
    print(f"\n[4/4] Parsing metadata...")
    meta_data = read_jsonl_gz(meta_file, sample_size=None)  # Load all metadata
    meta_df = pd.DataFrame(meta_data)
    print("Metadata columns:")
    print(meta_df.columns.tolist())
    print(f"✓ Loaded {len(meta_df):,} products")
    
    # Step 5: Process and combine
    print("\n[5/5] Processing and combining data...")
    
    # Combine title and review text
    reviews_df['review_text'] = (
        reviews_df['title'].fillna('').astype(str) + ' ' + 
        reviews_df['text'].fillna('').astype(str)
    ).str.strip()
    
    # Prepare metadata
    meta_df['product_title'] = meta_df['title'].fillna('').astype(str)
    meta_df['product_description'] = meta_df['description'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else str(x) if pd.notna(x) else ''
    )
    meta_df['category'] = meta_df['main_category'].fillna('Unknown')

    def extract_image_url(images):
        """
        images: list of dicts, each dict may have keys like 'hi_res', 'large', 'thumb'
        Returns the first valid URL it finds, preferring hi_res > large > thumb
        """
        if not isinstance(images, list):
            return None

        for img_dict in images:
            if not isinstance(img_dict, dict):
                continue
            for key in ["hi_res", "large", "thumb"]:
                url = img_dict.get(key)
                if isinstance(url, str) and url.startswith("http"):
                    return url
        return None

    meta_df["image_url"] = meta_df["images"].apply(extract_image_url)

    # Merge
    df = reviews_df.merge(
        meta_df[['parent_asin', 'product_title', 'product_description', 'category', 'image_url']],
        on='parent_asin',
        how='inner'
    )
    
    print(f"✓ Merged to {len(df):,} samples")
    
    # Create combined text field
    df['text'] = (
        df['product_title'].fillna('') + ' ' +
        df['product_description'].fillna('') + ' ' +
        df['review_text'].fillna('')
    ).str.strip()
    
    # Keep only needed columns
    df = df[['text', 'category', 'parent_asin', 'rating', 'image_url']].copy()
    
    # Remove empty texts
    df = df[df['text'].str.len() > 10].copy()
    
    # If we have too many, sample to requested size
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"✓ Final dataset: {len(df):,} samples")
    print(f"✓ Categories: {df['category'].nunique()}")
    print("\nCategory distribution:")
    print(df['category'].value_counts())
    
    # Save dataset
    output_path = f"{output_dir}/amazon_{category}_processed.csv"

    non_null_images = df["image_url"].notna().sum()
    print("\n[Image sanity check]")
    print("Non-null image URLs:", non_null_images)
    print(df["image_url"].head(5))
    if non_null_images == 0:
        raise RuntimeError(
            "No image URLs found. Metadata parsing failed — aborting image download."
        )
    df = download_images(df, image_dir=f"{output_dir}/images")

    df.to_csv(output_path, index=False)
    print(f"\n✓ Dataset saved to: {output_path}")
    
    # Save category mapping
    categories = sorted(df['category'].unique())
    category_map = {cat: idx for idx, cat in enumerate(categories)}
    map_path = f"{output_dir}/amazon_{category}_categories.json"
    with open(map_path, 'w') as f:
        json.dump(category_map, f, indent=2)
    print(f"✓ Category mapping saved to: {map_path}")
    
    print("\n" + "="*70)
    print("✓ DATASET READY!")
    print("="*70)
    print(f"\nYou can now run:")
    print(f"python main.py --data_path {output_path} --model_type text --num_epochs 3 --run_baseline")
    print()
    
    return output_path

def download_images(df, image_dir, max_images=None):
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        url = row['image_url']
        asin = row['parent_asin']

        if pd.isna(url):
            image_paths.append(None)
            continue

        out_path = image_dir / f"{asin}.jpg"

        if not out_path.exists():
            try:
                r = requests.get(url, timeout=10)
                img = Image.open(BytesIO(r.content)).convert("RGB")
                img.save(out_path)
            except Exception:
                image_paths.append(None)
                continue

        image_paths.append(str(out_path))

    df['image_path'] = image_paths
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Amazon Reviews 2023 dataset')
    parser.add_argument('--category', type=str, default='All_Beauty',
                       help='Category name (default: All_Beauty)')
    parser.add_argument('--sample_size', type=int, default=10000,
                       help='Number of samples (default: 10000)')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory (default: data)')
    
    args = parser.parse_args()
    
    try:
        download_amazon_dataset(
            category=args.category,
            sample_size=args.sample_size,
            output_dir=args.output_dir
        )
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)