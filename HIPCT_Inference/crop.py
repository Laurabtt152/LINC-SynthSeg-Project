import nibabel as nib

img = nib.load("/Users/lauraboettcher/Downloads/I74_level4/sub-I74_320um.nii.gz")
data = img.get_fdata()

crop = data[100:356, 100:356, 100:356]

crop_img = nib.Nifti1Image(crop, img.affine, img.header)
nib.save(crop_img, "sub-I74_320um_crop.nii.gz")