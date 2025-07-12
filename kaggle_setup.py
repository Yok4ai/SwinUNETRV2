import os
import glob
import json
import subprocess
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import re


def prepare_brats_data(input_dir, output_dir, dataset="brats2023"):
    """Prepare BraTS data and create dataset.json file."""
    os.makedirs(output_dir, exist_ok=True)
    file_paths = glob.glob(os.path.join(input_dir, '*'))
    file_paths.sort()
    file_names = [os.path.basename(path) for path in file_paths]
    file_names.sort()

    t1c, t1n, t2f, t2w, label = [], [], [], [], []

    for i in range(len(file_paths)):
        if dataset == "brats2021":
            t1c.append(os.path.join(file_paths[i], file_names[i] + '-t1c.nii', file_names[i] + '-t1c.nii'))
            t1n.append(os.path.join(file_paths[i], file_names[i] + '-t1n.nii', file_names[i] + '-t1n.nii'))
            t2f.append(os.path.join(file_paths[i], file_names[i] + '-t2f.nii', file_names[i] + '-t2f.nii'))
            t2w.append(os.path.join(file_paths[i], file_names[i] + '-t2.nii', file_names[i] + '-t2.nii'))
            label.append(os.path.join(file_paths[i], file_names[i] + '-seg.nii', file_names[i] + '-seg.nii'))
        else:
            t1c.append(os.path.join(file_paths[i], file_names[i] + '-t1c.nii'))
            t1n.append(os.path.join(file_paths[i], file_names[i] + '-t1n.nii'))
            t2f.append(os.path.join(file_paths[i], file_names[i] + '-t2f.nii'))
            t2w.append(os.path.join(file_paths[i], file_names[i] + '-t2w.nii'))
            label.append(os.path.join(file_paths[i], file_names[i] + '-seg.nii'))

    file_list = []
    for i in range(len(file_paths)):
        file_list.append({
            "image": [t1c[i], t1n[i], t2f[i], t2w[i]],
            "label": label[i]
        })

    file_json = {
        "training": file_list
    }

    output_json_path = os.path.join(output_dir, "dataset.json")
    with open(output_json_path, 'w') as json_file:
        json.dump(file_json, json_file, indent=4)

    print(f"Created dataset.json at: {output_json_path}")
    return output_dir

def setup_kaggle_notebook(dataset="brats2023"):
    """Setup the Kaggle notebook environment and prepare data for the specified dataset."""
    if dataset == "brats2021":
        input_dir = '/kaggle/input/brats21'
    else:
        input_dir = '/kaggle/input/brats2023-part-1'
    output_dir = '/kaggle/working'
    prepare_brats_data(input_dir, output_dir, dataset)
    return output_dir

if __name__ == "__main__":
    output_dir = setup_kaggle_notebook()
    print(f"Setup complete. Dataset prepared in: {output_dir}") 