import torch
from monai.transforms import MapTransform

class ConvertLabels(MapTransform):
    """
    Convert labels to multi channels based on BRATS 2023 classes:
    label 1 is Necrotic Tumor Core (NCR)
    label 2 is Edema (ED)
    label 3 is Enhancing Tumor (ET)
    label 0 is everything else (non-tumor)
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # Tumor Core (TC) = NCR + Enhancing Tumor (ET)
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            # Whole Tumor (WT) = NCR + Edema + Enhancing Tumor
            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 2), d[key] == 3))
            # Enhancing Tumor (ET) = Enhancing Tumor (label 3)
            result.append(d[key] == 3)
            # Stack along channel dimension (1) instead of creating a new dimension
            d[key] = torch.stack(result, dim=1).float()
        return d