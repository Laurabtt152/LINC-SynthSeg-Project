import numpy as np
import nibabel as nib
import scipy.ndimage 
import os

# Load the NextBrain segmentation file
nii_file_path = "/home/lboettcher/Downloads/sub-100263562592_seg-nextbrain_side-right_dseg.nii"
nii_data = nib.load(nii_file_path)
segmentation = nii_data.get_fdata()

# Define hierarchical mean intensity values based on the provided formula
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

# Create vector of means for the 17 tissue types
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

# Map segmentation labels to tissue types (Step 1)
relabeled_tissue_map = np.clip(segmentation, 0, 16).astype(int)

# Assign mean intensities to voxels (Step 4)
fake_image = cheating_means[relabeled_tissue_map]

# Add noise (Step 5)
noise_std = np.random.uniform(15, 25)
fake_image += np.random.normal(0, noise_std, fake_image.shape)

# Apply Gaussian blur (Step 6)
blurred_image = scipy.ndimage.gaussian_filter(fake_image, sigma=0.3)

# Generate segmentation masks according to ASEG-like protocol
cortex_mask = (segmentation > 1000).astype(int) * 3
white_matter_mask = np.isin(segmentation, [7, 68, 120, 130, 199, 100, 161, 208, 209, 102, 174, 254]).astype(int) * 2
thalamic_mask = np.isin(segmentation, [276, 444, 424, 400, 274, 578]).astype(int) * 4
thalamic_mask[segmentation == 484] = 0  # Subtract LGN
thalamic_mask[segmentation == 254] = 0  # Subtract Reticular
pallidum_mask = np.isin(segmentation, [119, 206]).astype(int) * 5
putamen_mask = np.isin(segmentation, [79, 349]).astype(int) * 6
caudate_accumbens_mask = np.isin(segmentation, [48, 118, 393, 101, 184]).astype(int) * 7
cerebellar_gray_mask = np.isin(segmentation, [595, 597]).astype(int) * 8

# Combine masks to create final segmentation
synthseg_labels = cortex_mask + white_matter_mask + thalamic_mask + pallidum_mask + putamen_mask + caudate_accumbens_mask + cerebellar_gray_mask

# Define output directory
output_dir = "/home/lboettcher/Downloads/"
os.makedirs(output_dir, exist_ok=True)

# Save the processed image as a NIfTI file
processed_nii = nib.Nifti1Image(blurred_image, nii_data.affine, nii_data.header)
nib.save(processed_nii, os.path.join(output_dir, "generated_nextbrain_image.nii.gz"))

# Save the segmentation mask as a NIfTI file
segmentation_nii = nib.Nifti1Image(synthseg_labels, nii_data.affine, nii_data.header)
nib.save(segmentation_nii, os.path.join(output_dir, "generated_synthseg_labels.nii.gz"))