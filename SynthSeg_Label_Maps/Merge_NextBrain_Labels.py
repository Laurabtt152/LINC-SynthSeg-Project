import numpy as np
import nibabel as nib
import random
import yaml
from pathlib import Path

def load_nifti(path):
    return nib.load(str(path)).get_fdata().astype(np.int32)

def save_nifti(path, data, ref_path):
    ref_img = nib.load(str(ref_path))
    new_img = nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header)
    nib.save(new_img, str(path))

def load_label_mapping(yaml_path):
    with open(yaml_path, 'r') as f:
        mapping = yaml.safe_load(f)
    # Assume the YAML maps NextBrain labels to integers [0–16]
    return {int(k): v for k, v in mapping.items()}

def apply_label_mapping(label_map, mapping_dict):
    mapped = np.zeros_like(label_map)
    for original_label, new_label in mapping_dict.items():
        mapped[label_map == original_label] = new_label
    return mapped

def get_region_label_ids(region):
    # These should match what's defined in your labeling schema
    if region == "left":
        return list(range(1000, 2000))  # Example: left hemisphere SynthSeg labels
    elif region == "right":
        return list(range(2000, 3000))
    elif region == "cerebellum":
        return [7, 45, 46, 47, 595, 597]  # Adjust as needed
    elif region == "brainstem":
        return [16, 170, 173, 174, 175]
    else:
        return []

def prepare_subject_sample(
    subject_id,
    synthseg_path,
    nextbrain_path,
    ontology_yaml,
    delete_probs={"left": 0.5, "right": 0.5, "cerebellum": 0.3, "brainstem": 0.3}
):
    # Load labels
    synthseg = load_nifti(synthseg_path)
    nextbrain = load_nifti(nextbrain_path)
    label_map = load_label_mapping(ontology_yaml)
    
    # Merge nextbrain labels
    merged_nextbrain = apply_label_mapping(nextbrain, label_map)

    deleted_labels = []

    for region, prob in delete_probs.items():
        if random.random() < prob:
            labels = get_region_label_ids(region)
            deleted_labels.extend(labels)
            merged_nextbrain[np.isin(merged_nextbrain, labels)] = 0
            synthseg[np.isin(synthseg, labels)] = 0

    return {
        "synthseg_labels": synthseg,
        "merged_nextbrain_labels": merged_nextbrain,
        "deleted_labels": deleted_labels
    }
