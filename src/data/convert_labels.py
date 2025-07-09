import torch
from monai.transforms import MapTransform

class ConvertLabels(MapTransform):
    """
    Convert labels to multi channels based on BRATS 2021 or 2023 classes:
    For BRATS 2023:
        label 1 is Necrotic Tumor Core (NCR)
        label 2 is Edema (ED)
        label 3 is Enhancing Tumor (ET)
        label 0 is everything else (background)
    For BRATS 2021:
        label 1 is Necrotic Tumor Core (NCR)
        label 2 is Edema (ED)
        label 4 is Enhancing Tumor (ET)
        label 0 is everything else (background)
    """
    def __init__(self, keys, dataset="brats2023"):
        super().__init__(keys)
        self.dataset = dataset
        if dataset == "brats2021":
            self.et_label = 4
        else:
            self.et_label = 3

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # Tumor Core (TC) = NCR + Enhancing Tumor (ET)
            result.append(torch.logical_or(d[key] == 1, d[key] == self.et_label))
            # Whole Tumor (WT) = NCR + Edema + Enhancing Tumor
            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 2), d[key] == self.et_label))
            # Enhancing Tumor (ET)
            result.append(d[key] == self.et_label)
            d[key] = torch.stack(result, axis=0).float()
        return d