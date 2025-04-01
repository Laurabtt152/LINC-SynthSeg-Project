import numpy as np
import nibabel as nib
import scipy.ndimage
import random
import yaml
import os

def load_nifti(path):
    return nib.load(str(path))

def load_label_mapping(yaml_path):
    with open(yaml_path, 'r') as f:
        mapping = yaml.safe_load(f)
    return {int(k): v for k, v in mapping.items()}

def apply_label_mapping(label_map, mapping_dict):
    merged = np.zeros_like(label_map, dtype=int)
    for original_label, new_label in mapping_dict.items():
        merged[label_map == original_label] = new_label
    return merged

def get_region_label_ids(region):
    if region == "left":
        return list(range(1000, 2000))  # example SynthSeg-style left hemi labels
    elif region == "right":
        return list(range(2000, 3000))
    elif region == "cerebellum":
        return [595, 597]
    elif region == "brainstem":
        return [16, 170, 173, 174, 175]
    else:
        return []

def prepare_training_sample(nextbrain_path, ontology_yaml, output_dir=None):
    # Load NextBrain segmentation
    nii = load_nifti(nextbrain_path)
    seg = nii.get_fdata().astype(np.int32)

    # Step 1: Merge labels using YAML
    label_map = load_label_mapping(ontology_yaml)
    merged = apply_label_mapping(seg, label_map)

    # Step 2: Randomly delete brain regions
    deleted_labels = []
    for region, prob in {"left": 0.5, "right": 0.5, "cerebellum": 0.3, "brainstem": 0.3}.items():
        if random.random() < prob:
            region_labels = get_region_label_ids(region)
            deleted_labels.extend(region_labels)
            merged[np.isin(seg, region_labels)] = 0
            seg[np.isin(seg, region_labels)] = 0

    # Step 3: Generate synthetic intensity values
    MU_BG = 255 * np.random.rand(1)
    MU_WM = 255 * np.random.rand(1)
    MU_GM = 255 * np.random.rand(1)
    MU_WM_CEREBELLUM = MU_WM + 30 * np.random.rand(1)
    MU_GM_CEREBELLUM = MU_GM + 30 * np.random.rand(1)
    MU_CAUDATE = MU_GM + 30 * np.random.rand(1)
    MU_PUTAMEN = MU_GM + 30 * np.random.rand(1)
    MU_PALLIDUM = MU_GM + 30 * np.random.rand(1)
    mid = 0.5 * MU_WM + 0.5 * MU_GM
    delta = (MU_WM - MU_GM) / 16.0
    MU_TH_LATERAL = mid + 5 * delta * np.random.rand(1)
    MU_TH_MEDIAL = mid - 5 * delta * np.random.rand(1)
    MU_RN = MU_WM + 20 * delta * np.random.rand(1)
    MU_GM_BS = MU_WM - 2 * delta * np.random.rand(1)
    MU_WM_BS = MU_WM + 10 * delta * np.random.rand(1)
    MU_HYPO = mid - 5 * delta * np.random.rand(1)
    MU_MAM_BODY = MU_WM
    MU_DG_CEREBELLUM = mid - 2 * delta * np.random.rand(1)
    MU_WM_HIPPO = mid + 2 * delta * np.random.rand(1)

    cheating_means = np.zeros(17)
    cheating_means[0] = MU_BG
    cheating_means[1] = MU_WM
    cheating_means[2] = MU_GM
    cheating_means[3] = MU_WM_CEREBELLUM
    cheating_means[4] = MU_GM_CEREBELLUM
    cheating_means[5] = MU_CAUDATE
    cheating_means[6] = MU_PUTAMEN
    cheating_means[7] = MU_TH_LATERAL
    cheating_means[8] = MU_TH_MEDIAL
    cheating_means[9] = MU_PALLIDUM
    cheating_means[10] = MU_RN
    cheating_means[11] = MU_GM_BS
    cheating_means[12] = MU_WM_BS
    cheating_means[13] = MU_HYPO
    cheating_means[14] = MU_MAM_BODY
    cheating_means[15] = MU_DG_CEREBELLUM
    cheating_means[16] = MU_WM_HIPPO

    relabeled_map = np.clip(merged, 0, 16).astype(int)
    fake_image = cheating_means[relabeled_map]

    # Step 5: Add noise
    noise_std = np.random.uniform(15, 25)
    fake_image += np.random.normal(0, noise_std, fake_image.shape)

    # Step 6: Apply Gaussian blur
    blurred_image = scipy.ndimage.gaussian_filter(fake_image, sigma=0.3)

    # Step 7: Create ASEG-style segmentation
    cortex_mask = (seg > 1000).astype(int) * 3
    wm_mask = np.isin(seg, [7, 68, 120, 130, 199, 100, 161, 208, 209, 102, 174, 254]).astype(int) * 2
    thalamic_mask = np.isin(seg, [276, 444, 424, 400, 274, 578]).astype(int) * 4
    thalamic_mask[seg == 484] = 0
    thalamic_mask[seg == 254] = 0
    pallidum_mask = np.isin(seg, [119, 206]).astype(int) * 5
    putamen_mask = np.isin(seg, [79, 349]).astype(int) * 6
    caudate_mask = np.isin(seg, [48, 118, 393, 101, 184]).astype(int) * 7
    cerebellar_mask = np.isin(seg, [595, 597]).astype(int) * 8

    synthseg_labels = (
        cortex_mask + wm_mask + thalamic_mask +
        pallidum_mask + putamen_mask + caudate_mask + cerebellar_mask
    )

    # Optional saving
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        nib.save(nib.Nifti1Image(blurred_image, nii.affine, nii.header),
                 os.path.join(output_dir, "generated_nextbrain_image.nii.gz"))
        nib.save(nib.Nifti1Image(synthseg_labels, nii.affine, nii.header),
                 os.path.join(output_dir, "generated_synthseg_labels.nii.gz"))
        nib.save(nib.Nifti1Image(merged, nii.affine, nii.header),
                 os.path.join(output_dir, "merged_nextbrain_labels.nii.gz"))

    return {
        "synthetic_image": blurred_image,
        "synthseg_labels": synthseg_labels,
        "merged_nextbrain_labels": merged,
        "deleted_labels": deleted_labels
    }
