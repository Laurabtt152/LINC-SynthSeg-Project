# transforms_synthseg.py
# -----------------------------------------------------------------------------
# MONAI-compatible transforms for your SynthSeg training loop.
# Designed to match calls in train_synthseg.py:
#   - MapLabelsSynthSeg()
#   - ResizeTransform(keys=["label"], spatial_size=..., method="pad_crop")
#   - SynthSegd(params=utils.get_synth_params(...), patch_size=...)
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.utils import InterpolateMode

# -------------------------
# Label mapping (merged NextBrain → 7 classes + background)
# -------------------------

# Final contiguous classes
CLASS_BG = 0
CLASS_WM = 1
CLASS_THALAMUS = 2
CLASS_PALLIDUM = 3
CLASS_PUTAMEN = 4
CLASS_CAUDATE_ACC = 5
CLASS_CEREBELLAR_GM = 6
CLASS_CORTEX = 7
_CORTEX_RULE_THRESHOLD = 1000  # any raw id > 1000 → cortex


class MapLabelsSynthSeg(MapTransform):
    """
    Merge raw NextBrain labels to a compact set of classes used for training.
    - Background stays 0
    - Explicit mappings in label_mapping()
    - IDs > 1000 → cortex
    """

    def __init__(self, keys: KeysCollection = ("label",)):
        super().__init__(keys=keys)

    @staticmethod
    def label_mapping() -> Dict[int, int]:
        # Matches your first-pass grouping from generation scripts
        # (WM, cortex, thalamus, pallidum, putamen, caudate/acc, cerebellar GM)
        return {
            0: CLASS_BG,
            # White Matter
            7: CLASS_WM, 68: CLASS_WM, 120: CLASS_WM, 130: CLASS_WM, 199: CLASS_WM,
            100: CLASS_WM, 161: CLASS_WM, 208: CLASS_WM, 209: CLASS_WM,
            102: CLASS_WM, 174: CLASS_WM, 254: CLASS_WM,
            # Thalamus (minus LGN=484, Reticular=254)
            276: CLASS_THALAMUS, 444: CLASS_THALAMUS, 424: CLASS_THALAMUS,
            400: CLASS_THALAMUS, 274: CLASS_THALAMUS, 578: CLASS_THALAMUS,
            # Pallidum
            119: CLASS_PALLIDUM, 206: CLASS_PALLIDUM,
            # Putamen
            79: CLASS_PUTAMEN, 349: CLASS_PUTAMEN,
            # Caudate + Accumbens
            48: CLASS_CAUDATE_ACC, 118: CLASS_CAUDATE_ACC, 393: CLASS_CAUDATE_ACC,
            101: CLASS_CAUDATE_ACC, 184: CLASS_CAUDATE_ACC,
            # Cerebellar Gray
            595: CLASS_CEREBELLAR_GM, 597: CLASS_CEREBELLAR_GM,
        }

    def __call__(self, data):
        d = dict(data)
        m = self.label_mapping()
        for k in self.keys:
            lab = d[k]
            lab_np = lab if isinstance(lab, np.ndarray) else lab.numpy()
            out = np.zeros_like(lab_np, dtype=np.int16)

            # explicit LUT mapping (fast for typical neuro IDs)
            max_id = int(lab_np.max()) if lab_np.size else 0
            if max_id < 100000:
                lut = np.zeros(max_id + 1, dtype=np.int16)
                for src, dst in m.items():
                    if 0 <= src <= max_id:
                        lut[src] = dst
                out = lut[lab_np.astype(np.int64)]
            else:
                for src, dst in m.items():
                    out[lab_np == src] = dst

            # cortex rule
            cortex_mask = lab_np > _CORTEX_RULE_THRESHOLD
            out[cortex_mask] = CLASS_CORTEX

            d[k] = out  # numpy array (MONAI will tensorize later)
        return d


# -------------------------
# Resize (pad/crop) helper
# -------------------------

class ResizeTransform(MapTransform):
    """
    Very small, dependency-light pad/crop to a target spatial size for label volumes.
    Assumes labels are (C, D, H, W) or (D, H, W). Uses nearest-neighbor behavior.
    method: "pad_crop" (centered) or None (no-op)
    """

    def __init__(
        self,
        keys: KeysCollection = ("label",),
        spatial_size: Optional[Tuple[int, int, int]] = None,
        method: Optional[str] = "pad_crop",
    ):
        super().__init__(keys=keys)
        self.spatial_size = spatial_size
        self.method = method

    @staticmethod
    def _ensure_c_first(x: np.ndarray) -> Tuple[np.ndarray, bool]:
        if x.ndim == 3:
            return x[None, ...], True
        return x, False

    @staticmethod
    def _center_pad_or_crop(x: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
        """x is (C, D, H, W); nearest-neighbor pad/crop centered."""
        C, D, H, W = x.shape
        d_t, h_t, w_t = out_shape
        # pad
        pad_d = max(0, d_t - D)
        pad_h = max(0, h_t - H)
        pad_w = max(0, w_t - W)
        if pad_d or pad_h or pad_w:
            pad = (
                (0, 0),
                (pad_d // 2, pad_d - pad_d // 2),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
            )
            x = np.pad(x, pad, mode="edge")
            C, D, H, W = x.shape
        # crop
        d0 = max(0, (D - d_t) // 2)
        h0 = max(0, (H - h_t) // 2)
        w0 = max(0, (W - w_t) // 2)
        return x[:, d0:d0 + d_t, h0:h0 + h_t, w0:w0 + w_t]

    def __call__(self, data):
        if self.spatial_size is None or self.method is None:
            return data
        d = dict(data)
        for k in self.keys:
            vol = d[k]
            vol_np = vol if isinstance(vol, np.ndarray) else vol.numpy()
            vol_np, added_channel = self._ensure_c_first(vol_np)
            vol_np = self._center_pad_or_crop(vol_np, self.spatial_size)
            if added_channel:
                vol_np = vol_np[0]
            d[k] = vol_np.astype(np.int16)
        return d


# -------------------------
# Label → synthetic image (lightweight)
# -------------------------

class SynthSegd(MapTransform):
    """
    Turn a label map into a synthetic MR-like image using simple, fast steps:
      - assign per-class Gaussian means (derived from params["target_labels"])
      - add noise, mild blur-like effect via percentile scaling in your pipeline
      - (optional) random patch extraction if patch_size is provided

    Expects/produces:
      - input dict has "label"
      - output dict adds "image" (float32, shape (1, D, H, W)) and keeps "label"

    This is intentionally lightweight to pair with your get_synth_params() choices. :contentReference[oaicite:2]{index=2}
    """

    def __init__(
        self,
        keys: KeysCollection = ("label",),
        params: Optional[Dict] = None,
        patch_size: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__(keys=keys)
        self.params = {} if params is None else params
        self.patch_size = patch_size

        # Pull a reproducible set of per-class base intensities
        # Based loosely on your get_synth_params ranges; tweak as needed. :contentReference[oaicite:3]{index=3}
        self.target_labels: List[int] = list(self.params.get("target_labels", []))
        if not self.target_labels:
            # Default to [1..7] if not provided
            self.target_labels = [1, 2, 3, 4, 5, 6, 7]

        rng = np.random.RandomState(0)
        base = {}
        for cls in self.target_labels:
            # spread means over [40..220] with small randomization
            base[cls] = rng.uniform(40, 220)
        self._base_means = base

        # Noise/SNR-ish knobs
        self._snr = float(self.params.get("snr", 15.0))
        self._gamma = float(self.params.get("gamma", 0.3))
        self._bias = float(self.params.get("bias", 3.0))

    @staticmethod
    def _ensure_cd_first(lbl_np: np.ndarray) -> Tuple[np.ndarray, bool]:
        if lbl_np.ndim == 3:
            return lbl_np[None, ...], True
        return lbl_np, False

    def _make_full_image(self, lbl_np: np.ndarray) -> np.ndarray:
        """
        lbl_np: (C, D, H, W) with C=1. Returns image (1, D, H, W) float32.
        """
        if lbl_np.shape[0] != 1:
            raise ValueError("Label must be single-channel (C=1).")
        lab = lbl_np[0].astype(np.int32)

        img = np.zeros_like(lab, dtype=np.float32)
        for cls, mu in self._base_means.items():
            img[lab == cls] = mu

        # Add Gaussian noise roughly controlled by "snr"
        std = max(1.0, np.mean(list(self._base_means.values())) / max(self._snr, 1.0))
        img = img + np.random.normal(0.0, std, size=img.shape).astype(np.float32)

        # Simple bias/gamma effects
        if self._bias > 0:
            # multiplicative smooth bias via super low-freq noise
            # (cheap stand-in; real pipeline might use B-splines)
            bf = np.random.normal(1.0, self._bias * 0.02, size=img.shape).astype(np.float32)
            img *= bf

        if self._gamma > 0:
            # crude gamma-like nonlinearity
            m = img - img.min()
            M = m.max() + 1e-6
            img = (m / M) ** (1.0 + self._gamma)
            img = img * 255.0

        # clip + normalize to [0,1]; your pipeline also calls ScaleIntensityd() later. :contentReference[oaicite:4]{index=4}
        img = np.clip(img, 0, 255).astype(np.float32)
        img = img / 255.0

        return img[None, ...]  # (1, D, H, W)

    @staticmethod
    def _random_patch(vol: np.ndarray, patch: Tuple[int, int, int]) -> np.ndarray:
        """vol is (C, D, H, W), returns centered random crop of size patch."""
        C, D, H, W = vol.shape
        pd, ph, pw = patch
        if pd > D or ph > H or pw > W:
            # fallback: center crop to min size
            pd, ph, pw = min(pd, D), min(ph, H), min(pw, W)
        d0 = np.random.randint(0, D - pd + 1) if D > pd else 0
        h0 = np.random.randint(0, H - ph + 1) if H > ph else 0
        w0 = np.random.randint(0, W - pw + 1) if W > pw else 0
        return vol[:, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]

    def __call__(self, data):
        d = dict(data)
        k = self.keys[0]  # "label"
        lbl = d[k]
        lbl_np = lbl if isinstance(lbl, np.ndarray) else lbl.numpy()
        lbl_np, added_channel = self._ensure_cd_first(lbl_np)

        # Make synthetic image
        img = self._make_full_image(lbl_np)  # (1, D, H, W)

        # Optional random patching (apply identically to image & label)
        if self.patch_size is not None:
            patch = self.patch_size
            img = self._random_patch(img, patch)
            lbl_np = self._random_patch(lbl_np, patch)

        if added_channel:
            lbl_np = lbl_np  # still (1, ...)

        d["image"] = img.astype(np.float32)
        d[k] = lbl_np.astype(np.int16)
        return d
