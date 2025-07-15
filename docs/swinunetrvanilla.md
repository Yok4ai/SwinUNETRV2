# Swin Transformers for Semantic Segmentation of Brain Tumors (Vanilla Implementation)

This document describes the baseline/vanilla SwinUNETR implementation as originally proposed, without any enhancements.

---

## 3.3 Loss Function

We use the **soft Dice loss function** [30] which is computed in a voxel-wise manner as:

$$\mathcal{L}(G, Y) = 1 - \frac{2}{J} \sum_{j=1}^{J} \frac{\sum_{i=1}^{I} G_{i,j} Y_{i,j}}{\sum_{i=1}^{I} G_{i,j}^2 + \sum_{i=1}^{I} Y_{i,j}^2}$$

where $I$ denotes voxels numbers; $J$ is classes number; $Y_{i,j}$ and $G_{i,j}$ denote the probability of output and one-hot encoded ground truth for class $j$ at voxel $i$, respectively.

---

## 3.4 Implementation Details

Swin UNETR is implemented using **PyTorch** and **MONAI** and trained on a DGX-1 cluster with 8 NVIDIA V100 GPUs. Table 1 details the configurations of Swin UNETR architecture, number of parameters and FLOPs. 

**Key configurations and training parameters:**

* **Learning Rate:** Set to 0.0008
* **Input Normalization:** All input images are normalized to have zero mean and unit standard deviation according to non-zero voxels
* **Patch Cropping:** Random patches of $128 \times 128 \times 128$ were cropped from 3D image volumes during training
* **Data Augmentation:**
    * **Random Axis Mirror Flip:** Applied with a probability of 0.5 for all 3 axes
    * **Random Per Channel Intensity Shift:** In the range $(-0.1, 0.1)$
    * **Random Scale of Intensity:** In the range $(0.9, 1.1)$ to input image channels
* **Batch Size:** 1 per GPU
* **Total Training:** All models were trained for a total of **800 epochs**
* **Learning Rate Scheduler:** Uses a **linear warmup** and **cosine annealing learning rate scheduler**
* **Inference:** Uses a **sliding window approach** with an overlapping of 0.7 for neighboring voxels

---

## 3.5 Dataset and Model Ensembling

The **BraTS challenge** aims to evaluate state-of-the-art methods for the semantic segmentation of brain tumors by providing a 3D MRI dataset with voxel-wise ground truth labels that are annotated by physicians [6,29,5,3,4].

**Dataset specifics (BraTS 2021):**

* **Training Dataset:** Includes **1251 subjects**, each with four 3D MRI modalities:
    * **Native T1-weighted (T1)**
    * **Post-contrast T1-weighted (T1Gd)**  
    * **T2-weighted (T2)**
    * **T2 Fluid-attenuated Inversion Recovery (T2-FLAIR)**

* **Preprocessing:** Images are rigidly aligned, resampled to a $1 \times 1 \times 1$ mm isotropic resolution, and skull-stripped
* **Input Image Size:** $240 \times 240 \times 155$
* **Data Collection:** Data were collected from multiple institutions using various MRI scanners

* **Annotations:** Include three tumor sub-regions:
    * The enhancing tumor
    * The peritumoral edema  
    * The necrotic and non-enhancing tumor core
    
    The annotations were combined into three nested sub-regions:
    * **Whole Tumor (WT)**
    * **Tumor Core (TC)**
    * **Enhancing Tumor (ET)**

* **Validation Dataset:** Consists of **219 cases** designed for intermediate model evaluations. Semantic segmentation labels corresponding to validation cases are not publicly available, and performance benchmarks were obtained by making submissions to the official server of BraTS 2021 challenge.

* **Testing Dataset:** Additional information regarding the testing dataset was not provided to participants.

**Model Training and Ensembling:**

* Models were trained on BraTS 2021 dataset with 1251 and 219 cases in the training and validation sets, respectively
* Used **five-fold cross-validation schemes** with a ratio of 80:20
* **No additional data** was used
* The **final result** was obtained with an **ensemble of 10 Swin UNETR models** to improve the performance and achieve a better consensus for all predictions
* The ensemble models were obtained from two separate five-fold cross-validation training runs

---

## 4 Results and Discussion

We have compared the performance of Swin UNETR in our internal cross validation split against the winning methodologies of previous years such as **SegResNet** [31], **nnU-Net** [19] and **TransBTS** [39]. The latter is a ViT-based approach which is tailored for the semantic segmentation of brain tumors.

### Table 2: Five-fold cross-validation benchmarks (mean Dice score)

| Fold | **Swin UNETR** |   |   | Avg. | **nnU-Net** |   |   | Avg. | **SegResNet** |   |   | Avg. | **TransBTS** |   |   | Avg. |
|------|:------:|:------:|:------:|:----:|:--------:|:--------:|:--------:|:----:|:---------:|:---------:|:---------:|:----:|:--------:|:--------:|:--------:|:----:|
|      | **ET** | **WT** | **TC** |      | **ET** | **WT** | **TC** |      | **ET** | **WT** | **TC** |      | **ET** | **WT** | **TC** |      |
| **Fold 1** | 0.876 | 0.929 | 0.914 | 0.906 | 0.866 | 0.921 | 0.902 | 0.896 | 0.867 | 0.924 | 0.907 | 0.899 | 0.856 | 0.910 | 0.897 | 0.883 |
| **Fold 2** | 0.908 | 0.938 | 0.919 | 0.921 | 0.899 | 0.933 | 0.919 | 0.917 | 0.900 | 0.933 | 0.915 | 0.916 | 0.885 | 0.919 | 0.903 | 0.902 |
| **Fold 3** | 0.891 | 0.931 | 0.919 | 0.913 | 0.886 | 0.929 | 0.914 | 0.910 | 0.884 | 0.927 | 0.917 | 0.909 | 0.866 | 0.903 | 0.898 | 0.889 |
| **Fold 4** | 0.890 | 0.937 | 0.920 | 0.915 | 0.886 | 0.927 | 0.914 | 0.909 | 0.888 | 0.921 | 0.916 | 0.908 | 0.868 | 0.910 | 0.901 | 0.893 |
| **Fold 5** | 0.891 | 0.934 | 0.917 | 0.914 | 0.880 | 0.929 | 0.917 | 0.909 | 0.878 | 0.930 | 0.912 | 0.906 | 0.867 | 0.915 | 0.893 | 0.892 |
| **Average** | **0.891** | **0.933** | **0.917** | **0.913** | **0.883** | **0.927** | **0.913** | **0.908** | **0.883** | **0.927** | **0.913** | **0.907** | **0.868** | **0.911** | **0.898** | **0.891** |

**Key findings from cross-validation:**

* The **Swin UNETR model outperforms** all competing approaches across all 5 folds and on average for all semantic classes (ET, WT, TC)
* Specifically, Swin UNETR outperforms the closest competing approaches by **0.7%, 0.6% and 0.4%** for ET, WT and TC classes respectively and on average **0.5%** across all classes in all folds
* The superior performance of Swin UNETR in comparison to other top performing models for brain tumor segmentation is mainly due to its capability of learning **multi-scale contextual information** in its hierarchical encoder via the self-attention modules and effective modeling of the **long-range dependencies**
* **nnU-Net and SegResNet** have competitive benchmarks in these experiments, with nnU-Net demonstrating a slightly better performance
* **TransBTS**, which is a ViT-based methodology, performs sub-optimally in comparison to other models. The sub-optimal performance of TransBTS could be attributed to its inefficient architecture in which the ViT is only utilized in the bottleneck as a standalone attention module, and without any connection to the decoder in different resolutions

### Table 3: BraTS 2021 Validation Dataset Benchmarks

| Metric | ET | WT | TC |
|:-------|:----:|:----:|:----:|
| **Dice Score** | 0.858 | 0.926 | 0.885 |
| **Hausdorff Distance (mm)** | 6.016 | 5.831 | 3.770 |

* The segmentation performance of Swin UNETR in the BraTS 2021 validation set shows that according to the official challenge results, our benchmarks (Team: **NVOptNet**) are considered as **one of the top-ranking methodologies** across more than **2000 submissions** during the validation phase, hence being the **first transformer-based model** to place competitively in BraTS challenges
* The segmentation outputs are **well-delineated** for all three sub-regions, consistent with quantitative benchmarks

### Table 4: BraTS 2021 Testing Dataset Benchmarks

| Metric | ET | WT | TC |
|:-------|:----:|:----:|:----:|
| **Dice Score** | 0.853 | 0.927 | 0.876 |
| **Hausdorff Distance (mm)** | 16.326 | 4.739 | 15.309 |

* The segmentation performance of Swin UNETR in the BraTS 2021 testing set shows that the segmentation performance of **ET and WT** are very similar to those of the validation benchmarks
* However, the segmentation performance of **TC** is decreased by **0.9%**

---

## 5 Conclusion

In this paper, we introduced **Swin UNETR** which is a novel architecture for semantic segmentation of brain tumors using multi-modal MRI images. Our proposed model has a U-shaped network design and uses a Swin transformer as the encoder and CNN-based decoder that is connected to the encoder via skip connections at different resolutions. 

We have validated the effectiveness of our approach in the **BraTS 2021 challenge**. Our model ranks among **top-performing approaches** in the validation phase and demonstrates **competitive performance** in the testing phase. We believe that Swin UNETR could be the foundation of a new class of transformer-based models with hierarchical encoders for the task of brain tumor segmentation.

---

**Note:** This document describes the vanilla/baseline Swin UNETR implementation. For enhanced features including 14 loss functions, adaptive scheduling, and local minima escape strategies, refer to the SwinUNETR++ documentation.