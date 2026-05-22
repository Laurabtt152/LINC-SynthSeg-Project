import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load image
img = nib.load("/Users/lauraboettcher/Downloads/sub-I74_320um_crop.nii.gz")
img_data = img.get_fdata()

# Load segmentation
seg = nib.load("./results/I74_crop_pred_seg.nii.gz")
seg_data = seg.get_fdata()

print("Image shape:", img_data.shape)
print("Seg shape:", seg_data.shape)
print("Labels:", np.unique(seg_data))

# Middle slice
z = img_data.shape[2] // 2

plt.figure(figsize=(14,6))

# Original HiP-CT
plt.subplot(1,2,1)
plt.imshow(img_data[:, :, z], cmap="gray")
plt.title("HiP-CT")
plt.axis("off")

# Overlay segmentation
plt.subplot(1,2,2)
plt.imshow(img_data[:, :, z], cmap="gray")
plt.imshow(seg_data[:, :, z], alpha=0.5)
plt.title("Segmentation Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()