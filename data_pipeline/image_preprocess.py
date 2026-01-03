import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

class ImagePreprocessor:
    def __init__(self, image_size=224, normalize=True):
        """
        Initialize image preprocessor with transforms

        Args:
            image_size: Target size for images (default 224 for ResNet/ViT)
            normalize: Whether to apply ImageNet normalization
        """
        self.image_size = image_size

        # Training transforms with augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ])

        # Validation/test transforms (no augmentation)
        self.eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # Add normalization if specified
        if normalize:
            # ImageNet mean and std
            normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            self.train_transform = transforms.Compose([
                self.train_transform,
                normalize_transform
            ])
            self.eval_transform = transforms.Compose([
                self.eval_transform,
                normalize_transform
            ])

    def load_image(self, image_path):
        """
        Load and validate image file

        Args:
            image_path: Path to image file

        Returns:
            PIL Image or None if invalid
        """
        # Guard against missing / NaN image paths
        try:
            import pandas as _pd
        except Exception:
            _pd = None

        if image_path is None:
            return None

        if _pd is not None and _pd.isna(image_path):
            return None

        if isinstance(image_path, float):
            # float values (e.g., nan) are invalid paths
            return None

        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def process_image(self, image_path, train=True):
        """
        Load and transform a single image

        Args:
            image_path: Path to image
            train: Whether to use training transforms

        Returns:
            Transformed image tensor or None
        """
        # Load image safely; load_image returns None for invalid/missing paths
        img = self.load_image(image_path)
        if img is None:
            return None

        transform = self.train_transform if train else self.eval_transform
        return transform(img)

    def verify_images(self, df, image_dir, image_column='image_id', ext='.jpg'):
        """
        Verify that all images exist and are valid

        Args:
            df: Dataframe with image references
            image_dir: Directory containing images
            image_column: Column name with image IDs
            ext: Image file extension

        Returns:
            Filtered dataframe with only valid images
        """
        print("Verifying images...")

        valid_indices = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image_id = row[image_column]
            # Skip missing image ids
            try:
                import pandas as _pd
            except Exception:
                _pd = None

            if image_id is None or (_pd is not None and _pd.isna(image_id)):
                continue

            image_path = os.path.join(image_dir, f"{image_id}{ext}")

            if os.path.exists(image_path):
                img = self.load_image(image_path)
                if img is not None:
                    valid_indices.append(idx)

        df_valid = df.loc[valid_indices].reset_index(drop=True)

        print(f"Valid images: {len(df_valid)} / {len(df)}")
        print(f"Invalid images: {len(df) - len(df_valid)}")

        return df_valid

    def create_image_paths_column(self, df, image_dir, image_column='image_id', ext='.jpg'):
        """
        Add full image paths to dataframe

        Args:
            df: Input dataframe
            image_dir: Directory containing images
            image_column: Column with image IDs
            ext: File extension

        Returns:
            Dataframe with 'image_path' column added
        """
        df['image_path'] = df[image_column].apply(
            lambda x: os.path.join(image_dir, f"{x}{ext}")
        )
        return df


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(image_size=224)

    # Load data and verify images
    # df = pd.read_csv('train.csv')
    # df = preprocessor.create_image_paths_column(df, 'images/', 'product_id')
    # df_valid = preprocessor.verify_images(df, 'images/', 'product_id')

    # Test single image
    # img_tensor = preprocessor.process_image('images/example.jpg', train=True)
    # print(f"Image shape: {img_tensor.shape}")  # Should be [3, 224, 224]

    pass