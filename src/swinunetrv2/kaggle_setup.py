import os
import glob
import json
import subprocess
import sys
import matplotlib.pyplot as plt

def setup_kaggle_environment():
    """Setup the required environment for Kaggle."""
    # Install MONAI with specific dependencies
    subprocess.check_call([
        sys.executable, "-c",
        "import monai" if subprocess.run([sys.executable, "-c", "import monai"]).returncode == 0
        else "import sys; subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'monai-weekly[nibabel, tqdm, einops]'])"
    ])
    
    # Install matplotlib if not present
    subprocess.check_call([
        sys.executable, "-c",
        "import matplotlib" if subprocess.run([sys.executable, "-c", "import matplotlib"]).returncode == 0
        else "import sys; subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'matplotlib'])"
    ])
    
    # Install einops
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "einops"])
    
    # Set matplotlib to inline mode
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.style.use('seaborn')
    
    # Print installed versions
    print("Environment setup complete. Installed packages:")
    subprocess.check_call([sys.executable, "-m", "pip", "list", "|", "grep", "-E", "monai|matplotlib|einops"])

def prepare_brats_data(input_dir, output_json_path):
    """Prepare BraTS data and create dataset.json file."""
    # Get sorted file paths and file names
    file_paths = glob.glob(os.path.join(input_dir, '*'))
    file_paths.sort()
    
    file_names = [os.path.basename(path) for path in file_paths]
    file_names.sort()
    
    # Initialize lists for different MRI modalities and segmentation labels
    t1c, t1n, t2f, t2w, label = [], [], [], [], []
    
    # Populate the lists with file paths
    for i in range(len(file_paths)):
        t1c.append(os.path.join(file_paths[i], file_names[i] + '-t1c.nii'))
        t1n.append(os.path.join(file_paths[i], file_names[i] + '-t1n.nii'))
        t2f.append(os.path.join(file_paths[i], file_names[i] + '-t2f.nii'))
        t2w.append(os.path.join(file_paths[i], file_names[i] + '-t2w.nii'))
        label.append(os.path.join(file_paths[i], file_names[i] + '-seg.nii'))
    
    # Store in a dictionary with combined image modalities and separate label
    file_list = []
    for i in range(len(file_paths)):
        file_list.append({
            "image": [t1c[i], t1n[i], t2f[i], t2w[i]],  # Combine modalities into one "image" field
            "label": label[i]
        })
    
    file_json = {
        "training": file_list
    }
    
    # Save to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(file_json, json_file, indent=4)
    
    return output_json_path

def setup_kaggle_notebook():
    """Setup the Kaggle notebook environment and prepare data."""
    # Setup environment
    setup_kaggle_environment()
    
    # Prepare data
    input_dir = '/kaggle/input/brats2023-part-1'
    output_json_path = '/kaggle/working/dataset.json'
    
    dataset_json_path = prepare_brats_data(input_dir, output_json_path)
    
    return dataset_json_path

if __name__ == "__main__":
    dataset_json_path = setup_kaggle_notebook()
    print(f"Dataset JSON created at: {dataset_json_path}") 