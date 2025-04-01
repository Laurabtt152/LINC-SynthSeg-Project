# 🧠 NextBrain Sample Preparation

This folder contains scripts for preparing synthetic neuroimaging data based on the **NextBrain** segmentation labels and converting them into **SynthSeg-style training samples**.

---

## 🚨 Important Note

The main file `Prepare_Subject_Sample.py` contains an integrated function that is **still under testing**. It combines logic from two separate scripts:

- `NextBrain.py`: A script that generates synthetic intensity images from relabeled NextBrain segmentation maps using a 17-tissue-type hierarchy, adding Gaussian noise and blur. A sample output can be found here: https://drive.google.com/drive/folders/1ITmiVhHr2_nZ4_HS8NcH0kkPeFE9EvMo?usp=sharing 
- `Merge_NextBrain_Labels`: A function that loads SynthSeg and NextBrain labels, merges the NextBrain labels using a YAML-based ontology, and **randomly deletes anatomical regions** (e.g., left hemisphere, brainstem) for training data augmentation.

The integrated version attempts to unify both pipelines by:
- Randomly deleting high-level brain regions.
- Merging NextBrain fine-grained labels into broader tissue classes.
- Generating fake MRI-like intensity images from those classes.
- Producing SynthSeg-style segmentation maps.

