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

    file_list = []
    if dataset == "brats2021":
        # 2021: /kaggle/input/brats21/BraTS21/BraTS-GLI-xxxxx-xxx/BraTS-GLI-xxxxx-xxx-t1c.nii
        case_dirs = glob.glob(os.path.join(input_dir, '*'))
        case_dirs = [d for d in case_dirs if os.path.isdir(d)]
        case_dirs.sort()
        for case_dir in case_dirs:
            case_id = os.path.basename(case_dir)
            t1c = os.path.join(case_dir, f"{case_id}-t1c.nii")
            t1n = os.path.join(case_dir, f"{case_id}-t1n.nii")
            t2f = os.path.join(case_dir, f"{case_id}-t2f.nii")
            t2w = os.path.join(case_dir, f"{case_id}-t2w.nii")
            label = os.path.join(case_dir, f"{case_id}-seg.nii")
            file_list.append({
                "image": [t1c, t1n, t2f, t2w],
                "label": label
            })
    else:
        # 2023: /kaggle/input/brats2023-part-1/BraTS-GLI-xxxxx-xxx-t1c.nii (no subfolder)
        file_paths = glob.glob(os.path.join(input_dir, '*'))
        file_paths.sort()
        file_names = [os.path.basename(path) for path in file_paths]
        file_names.sort()
        t1c, t1n, t2f, t2w, label = [], [], [], [], []
        for i in range(len(file_paths)):
            t1c.append(os.path.join(file_paths[i], file_names[i] + '-t1c.nii'))
            t1n.append(os.path.join(file_paths[i], file_names[i] + '-t1n.nii'))
            t2f.append(os.path.join(file_paths[i], file_names[i] + '-t2f.nii'))
            t2w.append(os.path.join(file_paths[i], file_names[i] + '-t2w.nii'))
            label.append(os.path.join(file_paths[i], file_names[i] + '-seg.nii'))
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
    """Setup the Kaggle notebook environment and prepare data."""
    if dataset == "brats2021":
        input_dir = '/kaggle/input/brats21/BraTS21'
    else:
        input_dir = '/kaggle/input/brats2023-part-1'
    output_dir = '/kaggle/working'
    prepare_brats_data(input_dir, output_dir, dataset=dataset)
    return output_dir

if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "brats2023"
    output_dir = setup_kaggle_notebook(dataset)
    print(f"Setup complete. Dataset prepared in: {output_dir}") 