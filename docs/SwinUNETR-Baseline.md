# Swin Transformers for Semantic Segmentation of Brain Tumors

## 3.3 Loss Function

The **soft Dice loss function** [30] is employed, computed in a voxel-wise manner as:

$\mathcal{L}(G, Y) = 1 - \frac{2}{J} \sum_{j=1}^{J} \frac{\sum_{i=1}^{I} G_{i,j} Y_{i,j}}{\sum_{i=1}^{I} G_{i,j}^2 + \sum_{i=1}^{I} Y_{i,j}^2}$

Here, $I$ represents the number of voxels, $J$ is the number of classes, $Y_{i,j}$ denotes the **probability of output** for class $j$ at voxel $i$, and $G_{i,j}$ signifies the **one-hot encoded ground truth** for class $j$ at voxel $i$.

---

## 3.4 Implementation Details

The **Swin UNETR** model is implemented using **PyTorch** and **MONAI**, and trained on a DGX-1 cluster equipped with 8 NVIDIA V100 GPUs.

**Key configurations and training parameters:**

* **Learning Rate:** Set to 0.0008.
* **Input Normalization:** All input images are normalized to have zero mean and unit standard deviation based on non-zero voxels.
* **Patch Cropping:** Random patches of $128 \times 128 \times 128$ voxels are cropped from 3D image volumes during training.
* **Data Augmentation:**
    * **Random Axis Mirror Flip:** Applied with a probability of 0.5 for all three axes.
    * **Random Per Channel Intensity Shift:** In the range $(-0.1, 0.1)$.
    * **Random Scale of Intensity:** In the range $(0.9, 1.1)$ to input image channels.
* **Batch Size:** 1 per GPU.
* **Epochs:** Models were trained for a total of **800 epochs**.
* **Learning Rate Scheduler:** Utilizes a **linear warmup** followed by a **cosine annealing learning rate scheduler**.
* **Inference:** A **sliding window approach** with an overlap of 0.7 for neighboring voxels is used.

---

## 3.5 Dataset and Model Ensembling

The **BraTS challenge** aims to evaluate semantic segmentation methods for brain tumors using a 3D MRI dataset with voxel-wise ground truth labels annotated by physicians [6, 29, 5, 3, 4].

**Dataset specifics (BraTS 2021):**

* **Training Dataset:** Includes **1251 subjects**.
* **MRI Modalities:** Each subject has four 3D MRI modalities:
    * Native T1-weighted (T1)
    * Post-contrast T1-weighted (T1Gd)
    * T2-weighted (T2)
    * T2 Fluid-attenuated Inversion Recovery (T2-FLAIR)
* **Preprocessing:** Images are rigidly aligned, resampled to a $1 \times 1 \times 1$ mm isotropic resolution, and skull-stripped.
* **Input Image Size:** $240 \times 240 \times 155$.
* **Annotations:** Three tumor sub-regions are annotated: enhancing tumor, peritumoral edema, and necrotic and non-enhancing tumor core. These are combined into three nested sub-regions:
    * **Whole Tumor (WT)**
    * **Tumor Core (TC)**
    * **Enhancing Tumor (ET)**
* **Validation Dataset:** Consists of **219 cases** for intermediate model evaluations, with ground truth labels not publicly available. Submissions were made to the organizers' server for evaluation.
* **Testing Dataset:** Additional information was not provided to participants, with evaluations also via server submission.

**Model Training and Ensembling:**

* Models were trained on the BraTS 2021 dataset (1251 training cases, 219 validation cases).
* A **five-fold cross-validation scheme** with an 80:20 ratio was used. No additional data was utilized.
* The **final result** was obtained from an **ensemble of 10 Swin UNETR models** to enhance performance and achieve better prediction consensus. These ensemble models were derived from two separate five-fold cross-validation training runs.

---

## 4 Results and Discussion

The performance of Swin UNETR was compared against previous winning methodologies like SegResNet [31], nnU-Net [19], and TransBTS [39] (a ViT-based approach for brain tumor segmentation).

**Table 2: Five-fold cross-validation benchmarks (mean Dice score)**

| Metric           | Swin UNETR (Avg.) | nnU-Net (Avg.) | SegResNet (Avg.) | TransBTS (Avg.) |
| :--------------- | :---------------- | :------------- | :--------------- | :-------------- |
| **Dice Score ET** | 0.891             | 0.883          | 0.883            | 0.868           |
| **Dice Score WT** | 0.933             | 0.927          | 0.927            | 0.911           |
| **Dice Score TC** | 0.917             | 0.913          | 0.913            | 0.898           |
| **Dice Score Avg.**| **0.913** | 0.908          | 0.907            | 0.891           |

**Key findings from cross-validation:**

* The **Swin UNETR model consistently outperforms** all competing approaches across all 5 folds and on average for all semantic classes (ET, WT, TC).
* Specifically, Swin UNETR shows superior performance compared to the closest competitors by **0.7% for ET, 0.6% for WT, and 0.4% for TC**, averaging **0.5% across all classes** in all folds.
* This superior performance is attributed to Swin UNETR's ability to learn **multi-scale contextual information** via hierarchical encoder and self-attention modules, effectively modeling **long-range dependencies**.
* **nnU-Net and SegResNet** show competitive benchmarks, with nnU-Net performing slightly better.
* **TransBTS** performs sub-optimally, possibly due to its architecture where the ViT is only used in the bottleneck as a standalone attention module without connections to the decoder at different resolutions.

**Table 3: BraTS 2021 Validation Dataset Benchmarks**

| Metric                     | ET      | WT      | TC      |
| :------------------------- | :------ | :------ | :------ |
| **Dice Score** | 0.858   | 0.926   | 0.885   |
| **Hausdorff Distance (mm)**| 6.016   | 5.831   | 3.770   |

* Swin UNETR's benchmarks (Team: NVOptNet) on the BraTS 2021 validation set placed it as **one of the top-ranking methodologies** among over 2000 submissions, making it the **first transformer-based model to perform competitively** in BraTS challenges.
* Figure 3 illustrates typical segmentation outputs from Swin UNETR for several validation cases, showing **well-delineated segmentation** for all three sub-regions, consistent with quantitative benchmarks.

**Table 4: BraTS 2021 Testing Dataset Benchmarks**

| Metric                     | ET      | WT      | TC      |
| :------------------------- | :------ | :------ | :------ |
| **Dice Score** | 0.853   | 0.927   | 0.876   |
| **Hausdorff Distance (mm)**| 16.326  | 4.739   | 15.309  |

* On the BraTS 2021 testing set, the segmentation performance for **ET and WT** is very similar to validation benchmarks.
* However, the segmentation performance for **TC** decreased by 0.9%.