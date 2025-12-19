#!/bin/bash

# Setup script for Amazon Reviews 2023 dataset
# This downloads and prepares the dataset for training

echo "=========================================="
echo "Amazon Reviews 2023 Dataset Setup"
echo "=========================================="

# Create directories
echo "Creating directories..."
mkdir -p data
mkdir -p data/amazon_images
mkdir -p models_saved
mkdir -p logs
mkdir -p results
mkdir -p figures

# Install dependencies
echo "Installing dependencies..."
pip install datasets

# Available categories (choose one or more)
# All_Beauty, Electronics, Clothing_Shoes_and_Jewelry, Sports_and_Outdoors, 
# Home_and_Kitchen, Health_and_Household, Toys_and_Games, etc.

# Default: Start with a smaller category for testing
CATEGORY="All_Beauty"
SAMPLE_SIZE=10000  # Use smaller sample for quick testing

echo ""
echo "Downloading and processing dataset..."
echo "Category: $CATEGORY"
echo "Sample size: $SAMPLE_SIZE"
echo ""

# Run the loader
python data_pipeline/load_amazon_2023.py \
    --category $CATEGORY \
    --sample_size $SAMPLE_SIZE \
    --download_images \
    --output_dir data

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Dataset saved to: data/amazon_${CATEGORY}_processed.csv"
echo ""
echo "Next steps:"
echo "1. For quick test (text-only, no images):"
echo "   python main.py --data_path data/amazon_${CATEGORY}_processed.csv --model_type text --num_epochs 3 --run_baseline"
echo ""
echo "2. For full multimodal training:"
echo "   python main.py --data_path data/amazon_${CATEGORY}_processed.csv --image_dir data/amazon_images --model_type multimodal --num_epochs 10"
echo ""
echo "3. For active learning:"
echo "   python main.py --data_path data/amazon_${CATEGORY}_processed.csv --image_dir data/amazon_images --model_type multimodal --use_active_learning"
echo ""
echo "To use a different category, edit this script and change CATEGORY variable."
echo "Available categories: All_Beauty, Electronics, Clothing_Shoes_and_Jewelry, etc."