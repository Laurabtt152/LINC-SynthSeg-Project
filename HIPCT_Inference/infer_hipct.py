#!/usr/bin/env python3
"""
Inference-only script for applying a trained SynthSeg/MONAI model to HiP-CT data.

Expected input:
    A real HiP-CT image volume already converted/extracted to a NIfTI file (.nii or .nii.gz),
    ideally at the resolution level closest to ~300um 

Example:
    python infer_hipct.py \
        --image /path/to/I74_level4_300um.nii.gz \
        --checkpoint ./results/model_best.pth \
        --output ./results/I74_pred_seg.nii.gz

"""

import argparse
import os
from pathlib import Path

import torch
import nibabel as nib

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
import utils_synthseg as utils


def load_model(checkpoint_path: str, device: torch.device, out_channels: int = 15):
    """
    Load the same BasicUNet architecture used in training.

    Supports:
    1. model_best.pth, which was saved as model.state_dict()
    2. checkpoint.pkl, which was saved as a dict containing "model_state_dict"
    """
    model = utils.get_model(out_channels=out_channels).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handles checkpoints saved from DistributedDataParallel.
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def run_inference(
    image_path: str,
    checkpoint_path: str,
    output_path: str,
    patch_size=(128, 128, 128),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    clip_min: float | None = None,
    clip_max: float | None = None,
):
    """
    Apply the trained model to one HiP-CT image volume and save an integer label map.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_path = str(image_path)
    checkpoint_path = str(checkpoint_path)
    output_path = str(output_path)
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Your trained model has BG + 14 foreground classes = 15 channels.
    out_channels = 15
    model = load_model(checkpoint_path, device=device, out_channels=out_channels)

    # For HiP-CT inference:
    # - Load the real image, not a label map.
    # - Keep orientation consistent with training.
    # - Normalize intensities.
    # You may need to tune clip_min/clip_max depending on HiP-CT intensity range.
    if clip_min is None or clip_max is None:
        intensity_transform = ScaleIntensityRanged(
            keys="image",
            a_min=0,
            a_max=65535,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
    else:
        intensity_transform = ScaleIntensityRanged(
            keys="image",
            a_min=clip_min,
            a_max=clip_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )

    infer_transforms = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        intensity_transform,
        EnsureTyped(keys="image", dtype=torch.float32, device=device),
    ])

    data = [{"image": image_path}]
    dataset = Dataset(data=data, transform=infer_transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            print(f"Input image shape: {tuple(image.shape)}")

            # Sliding-window inference prevents loading the whole prediction volume into
            # the network at once. Keep patch_size the same as training initially.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=patch_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                )

            # Convert channel logits to discrete class labels.
            seg = torch.argmax(logits, dim=1, keepdim=True).to(torch.uint8)
            print(f"Output segmentation shape: {tuple(seg.shape)}")
            print(f"Predicted labels present: {torch.unique(seg).detach().cpu().numpy()}")

            # Save manually with nibabel. This avoids MONAI SaveImaged metadata issues
            # that can happen with MetaTensor affine fields on some MONAI versions.
            seg_np = seg.detach().cpu().numpy()[0, 0].astype("uint8")
            ref_img = nib.load(image_path)
            out_img = nib.Nifti1Image(seg_np, affine=ref_img.affine, header=ref_img.header)
            out_img.set_data_dtype("uint8")
            nib.save(out_img, output_path)
            print(f"Saved segmentation to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Apply trained SynthSeg/MONAI model to HiP-CT NIfTI data.")
    parser.add_argument("--image", required=True, help="Path to HiP-CT NIfTI image, e.g. I74_level4_300um.nii.gz")
    parser.add_argument("--checkpoint", default="./results/model_best.pth", help="Path to model_best.pth or checkpoint.pkl")
    parser.add_argument("--output", required=True, help="Output segmentation path, e.g. ./results/I74_pred_seg.nii.gz")
    parser.add_argument("--patch-size", type=int, default=128, help="Cubic sliding-window patch size. Start with 128.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap.")
    parser.add_argument("--sw-batch-size", type=int, default=1, help="Sliding-window batch size.")
    parser.add_argument("--clip-min", type=float, default=None, help="Optional HiP-CT intensity lower clipping value.")
    parser.add_argument("--clip-max", type=float, default=None, help="Optional HiP-CT intensity upper clipping value.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        patch_size=(args.patch_size, args.patch_size, args.patch_size),
        overlap=args.overlap,
        sw_batch_size=args.sw_batch_size,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )
