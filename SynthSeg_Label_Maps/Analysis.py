import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean, directed_hausdorff
from sklearn.metrics import confusion_matrix

# Dice coefficient
def dice_coefficient(seg1, seg2):
    intersection = np.sum((seg1 > 0) & (seg2 > 0))
    return (2. * intersection) / (np.sum(seg1 > 0) + np.sum(seg2 > 0))

# Hausdorff distance
def hausdorff_distance(seg1, seg2):
    seg1_points = np.argwhere(seg1 > 0)
    seg2_points = np.argwhere(seg2 > 0)
    return max(directed_hausdorff(seg1_points, seg2_points)[0],
               directed_hausdorff(seg2_points, seg1_points)[0])

# PCA + Euclidean distance between principal components
def eigenvector_distance(seg1, seg2):
    seg1_points = np.argwhere(seg1 > 0)
    seg2_points = np.argwhere(seg2 > 0)
    pca1 = PCA(n_components=3).fit(seg1_points)
    pca2 = PCA(n_components=3).fit(seg2_points)
    return np.linalg.norm(pca1.components_ - pca2.components_)

# Composite comparison function
def compare_segmentations(seg1_path, seg2_path):
    seg1_img = nib.load(seg1_path)
    seg2_img = nib.load(seg2_path)
    
    seg1_data = seg1_img.get_fdata()
    seg2_data = seg2_img.get_fdata()
    
    assert seg1_data.shape == seg2_data.shape, "Segmentations must be the same shape"

    print("Dice Coefficient:", dice_coefficient(seg1_data, seg2_data))
    print("Hausdorff Distance:", hausdorff_distance(seg1_data, seg2_data))
    print("Eigenvector Distance:", eigenvector_distance(seg1_data, seg2_data))

# Example usage
if __name__ == "__main__":
    seg1_path = "/home/lboettcher/Downloads/sub-104893688989_seg-nextbrain_side-left_dseg.nii.gz"
    seg2_path = "/home/lboettcher/Downloads/generated_synthseg_labels_left.nii.gz"
    compare_segmentations(seg1_path, seg2_path)
