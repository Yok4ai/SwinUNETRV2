import os
import glob
import json
import subprocess
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import re

def setup_kaggle_environment():
    """Setup the Kaggle notebook environment."""
    # Install required packages
    packages = [
        "monai",
        "matplotlib",
        "einops",
        "torch",
        "torchvision",
        "pytorch-lightning",
        "nibabel",
        "scikit-learn",
        "tqdm",
        "numpy",
        "pandas"
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Print installed versions
    print("Environment setup complete. Installed packages:")
    result = subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode()
    # Filter for relevant packages
    relevant_packages = [line for line in result.split('\n') 
                        if any(pkg in line.lower() for pkg in ['monai', 'matplotlib', 'einops'])]
    print('\n'.join(relevant_packages))

def prepare_brats_data(input_dir, output_dir):
    """Prepare BraTS data for training."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all case directories
    case_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Create dataset.json structure
    dataset = {
        "name": "BraTS2023",
        "description": "BraTS 2023 Training Dataset",
        "reference": "https://www.synapse.org/#!Synapse:syn27046444/wiki/",
        "licence": "https://www.synapse.org/#!Synapse:syn27046444/wiki/",
        "tensorImageSize": "4D",
        "modality": {
            "0": "FLAIR",
            "1": "T1w",
            "2": "T1gd",
            "3": "T2w"
        },
        "labels": {
            "0": "background",
            "1": "NCR",
            "2": "ED",
            "3": "ET"
        },
        "training": []
    }
    
    # Process each case
    for case_dir in case_dirs:
        case_path = os.path.join(input_dir, case_dir)
        
        # Get image and label files
        image_files = sorted([f for f in os.listdir(case_path) if f.endswith('.nii.gz') and not f.endswith('seg.nii.gz')])
        label_file = [f for f in os.listdir(case_path) if f.endswith('seg.nii.gz')][0]
        
        # Create case entry
        case_entry = {
            "image": [os.path.join(case_path, f) for f in image_files],
            "label": os.path.join(case_path, label_file)
        }
        dataset["training"].append(case_entry)
    
    # Save dataset.json
    dataset_path = os.path.join(output_dir, "dataset.json")
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    return output_dir

def setup_kaggle_notebook():
    """Setup the Kaggle notebook environment and prepare data."""
    # Setup environment
    setup_kaggle_environment()
    
    # Prepare data
    input_dir = "/kaggle/input/brats2023-part-1"
    output_dir = "/kaggle/working"
    
    # Create dataset.json in working directory
    prepare_brats_data(input_dir, output_dir)
    
    return output_dir  # Return the directory path, not the file path

if __name__ == "__main__":
    output_dir = setup_kaggle_notebook()
    print(f"Setup complete. Dataset prepared in: {output_dir}") 