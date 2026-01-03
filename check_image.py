import os
import pandas as pd

CSV_PATH = "data/amazon_electronics_processed.csv"
IMAGE_COL = "image_path"

def normalize_path(p):
    if pd.isna(p):
        return None

    p = str(p).replace("\\", "/")

    # Fix duplicated prefixes
    if "data/images/data/images" in p:
        p = p.replace("data/images/data/images", "data/images")

    return p

def main():
    print("🔍 Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"✓ Loaded {len(df)} rows")

    missing_path = 0
    missing_file = 0
    valid = 0

    print("\n🧪 Checking image associations...")

    for _, row in df.iterrows():
        raw_path = row.get(IMAGE_COL)
        path = normalize_path(raw_path)

        if path is None:
            missing_path += 1
        elif not os.path.exists(path):
            missing_file += 1
        else:
            valid += 1

    print("\n📊 IMAGE ALIGNMENT REPORT")
    print("-------------------------")
    print(f"Total rows:         {len(df)}")
    print(f"Missing image path: {missing_path}")
    print(f"Missing image file: {missing_file}")
    print(f"Valid images:       {valid}")

    if valid == 0:
        print("\n❌ No valid images found — image download likely failed.")
    elif valid < len(df):
        print("\n⚠️ Dataset has partial image coverage.")
    else:
        print("\n✅ All rows have valid images. Multimodal is safe.")

if __name__ == "__main__":
    main()
