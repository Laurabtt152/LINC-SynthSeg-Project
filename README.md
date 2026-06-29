# LINC SynthSeg Project: High-Resolution Brain Segmentation with NextBrain

## Overview

This project investigates the use of synthetic deep learning training for high-resolution neuroanatomical segmentation. Building upon the original **SynthSeg** framework, this work incorporates the **NextBrain** histology-derived atlas to improve anatomical detail and enable segmentation of ultra-high-resolution ex vivo brain imaging, including **HiP-CT**.

Unlike traditional segmentation methods that rely on manually annotated MRI datasets, SynthSeg generates unlimited synthetic training data directly from anatomical label maps. This project extends that framework by replacing the original atlas labels with high-resolution NextBrain labels and evaluating whether these richer anatomical representations improve generalization to HiP-CT data.

The long-term goal is to develop a segmentation model capable of accurately labeling whole-brain HiP-CT volumes while remaining compatible with conventional MRI.

---

# Motivation

SynthSeg has demonstrated excellent performance on clinical and research MRI by training exclusively on synthetic images generated from anatomical label maps. However, the anatomical detail it can learn is fundamentally limited by the resolution of the atlases used during training.

The **NextBrain** atlas provides substantially finer anatomical information through histology-derived labels (~200–300 μm resolution) containing hundreds of neuroanatomical regions. This project investigates whether training SynthSeg using NextBrain-derived labels enables segmentation models to capture finer anatomical structures and generalize to ultra-high-resolution HiP-CT imaging.

---

# Repository Structure

```
LINC-SynthSeg-Project/
│
├── Anatomical_Hierarchy/
│   ├── Ontology parsing
│   ├── Anatomical hierarchy utilities
│   └── Label relationship processing
│
├── SynthSeg_Label_Maps/
│   ├── NextBrain label preprocessing
│   ├── Label remapping
│   └── Synthetic training label generation
│
├── Training_Code/
│   ├── MONAI implementation
│   ├── Model training
│   ├── Validation
│   └── Checkpoint management
│
├── HIPCT_Inference/
│   ├── HiP-CT preprocessing
│   ├── Sliding-window inference
│   └── Prediction generation
│
└── README.md
```

---

# Pipeline

## 1. NextBrain Label Preparation

The original NextBrain atlas contains hundreds of fine-grained anatomical labels. Training directly on every individual label is currently impractical, so these labels are merged into a unified set of anatomical classes suitable for deep learning.

This stage includes:

- Ontology parsing
- Anatomical hierarchy construction
- Label remapping
- Region filtering
- Generation of SynthSeg-compatible segmentation maps

Scripts for this stage are located in:

```
SynthSeg_Label_Maps/
```

---

## 2. Synthetic Image Generation

Following the SynthSeg paradigm, the network is trained entirely on synthetic MRI volumes generated from segmentation maps rather than manually annotated MRI scans.

The synthetic generation pipeline introduces randomized imaging characteristics including:

- Tissue intensity variation
- Gaussian noise
- Bias fields
- Image blurring
- Anatomical deformations

This approach enables virtually unlimited training data without requiring manual annotations.

---

## 3. Model Training

Training is implemented using **PyTorch** and **MONAI**.

The current implementation includes:

- BasicUNet architecture
- Dice loss optimization
- On-the-fly synthetic image generation
- RAS orientation normalization
- Automatic checkpointing
- Multi-GPU/distributed training support

The current model predicts:

- Background
- 14 anatomical classes

Training code is located in:

```
Training_Code/
```

---

## 4. HiP-CT Inference

After training, the network is applied to real HiP-CT image volumes.

The inference pipeline includes:

- Sliding-window inference
- Intensity normalization
- Model checkpoint loading
- Prediction generation
- NIfTI output export

Current experiments have focused on cropped HiP-CT volumes before scaling to full-brain inference.

Inference scripts are located in:

```
HIPCT_Inference/
```

---

# Current Status

## Completed

- ✅ NextBrain label preprocessing
- ✅ Anatomical hierarchy construction
- ✅ Label remapping pipeline
- ✅ Synthetic image generation
- ✅ MONAI training implementation
- ✅ Training on synthetic NextBrain-derived data
- ✅ HiP-CT inference pipeline
- ✅ Preliminary inference on real HiP-CT image crops

---

## In Progress

- Optimization of HiP-CT intensity normalization
- Evaluation on additional HiP-CT datasets
- Whole-brain HiP-CT inference
- Integration with high-resolution HiP-CT template construction

---

# Current Results

Training successfully converges on synthetic data generated from NextBrain-derived anatomical labels.

Preliminary inference demonstrates that the model is capable of producing anatomically plausible segmentations on real HiP-CT image volumes, providing encouraging evidence that synthetic training can generalize beyond conventional MRI.

Current work focuses on improving:

- Boundary accuracy
- Intensity normalization
- Segmentation robustness
- Whole-brain inference

---

# Future Directions

- Whole-brain HiP-CT segmentation
- Expansion to larger anatomical label sets
- Quantitative evaluation against manual annotations
- Integration with the LINC HiP-CT template
- Multi-resolution segmentation across MRI and HiP-CT
- Further refinement of NextBrain anatomical labels

---

# Dependencies

Major packages include:

- Python
- PyTorch
- MONAI
- NumPy
- nibabel
- matplotlib

Additional dependencies can be found in:

```
Training_Code/requirements.txt
```

---

# Acknowledgements

This project was completed as part of the **Large-scale Imaging of Neural Circuits (LINC)** initiative at the **Athinoula A. Martinos Center for Biomedical Imaging**, **Massachusetts General Hospital**, and **Harvard Medical School**.

The work builds upon the original **SynthSeg** framework while extending it with the **NextBrain** atlas to investigate high-resolution neuroanatomical segmentation for ultra-high-resolution brain imaging.
