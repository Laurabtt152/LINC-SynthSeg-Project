import numpy as np
import torch
import cornucopia as cc
from random import randrange
from typing import Sequence, Union

from monai.transforms import (
    MapTransform,
    RandomizableTransform,
    Resized,
    ResizeWithPadOrCropd,
)
from monai.config.type_definitions import KeysCollection
from monai.data.meta_tensor import MetaTensor

# -------------------------
# Compact class space (BG + 14)
# -------------------------
CLASS_BG   = 0  # Background + CSF + ventricles
CLASS_WM   = 1  # Cerebral white matter
CLASS_GM   = 2  # Cerebral gray matter (not cortex)
CLASS_CX   = 3  # Cerebral cortex (gray matter)
CLASS_CBWM = 4  # Cerebellar white matter
CLASS_CBCX = 5  # Cerebellar cortex (gray matter)
CLASS_TH   = 6  # Thalamus
CLASS_CDAC = 7  # Caudate + Accumbens
CLASS_PU   = 8  # Putamen
CLASS_PAL  = 9  # Pallidum
CLASS_BS   = 10 # Brain stem
CLASS_HIGM = 11 # Hippocampus GM
CLASS_HIWM = 12 # Hippocampus WM
CLASS_AM   = 13 # Amygdala
CLASS_DC   = 14 # Ventral DC

_CORTEX_RULE_THRESHOLD = 1000  # any raw id > 1000 → cortex

# -------------------------
# NEXT_TO_ASEG LUT (subset shown here; full list mirrors NextBrainLabels.py)
# -------------------------
NEXT_TO_ASEG = {
    # --- White matter ---
    7: CLASS_WM, 68: CLASS_WM, 120: CLASS_WM, 130: CLASS_WM, 199: CLASS_WM,
    100: CLASS_WM, 161: CLASS_WM, 208: CLASS_WM, 209: CLASS_WM, 102: CLASS_WM,
    174: CLASS_WM, 254: CLASS_WM,
    # --- Gray matter ---
    103: CLASS_GM, 108: CLASS_GM, 111: CLASS_GM, 117: CLASS_GM, 128: CLASS_GM,
    99: CLASS_GM, 113: CLASS_GM, 125: CLASS_GM, 157: CLASS_GM,
    # --- Hippocampal GM ---
    344: CLASS_HIGM, 369: CLASS_HIGM, 409: CLASS_HIGM, 567: CLASS_HIGM, 340: CLASS_HIGM, 563: CLASS_HIGM,
    419: CLASS_HIGM, 373: CLASS_HIGM, 572: CLASS_HIGM, 345: CLASS_HIGM, 370: CLASS_HIGM, 374: CLASS_HIGM,
    410: CLASS_HIGM, 420: CLASS_HIGM, 564: CLASS_HIGM, 573: CLASS_HIGM, 341: CLASS_HIGM, 568: CLASS_HIGM,
    371: CLASS_HIGM, 375: CLASS_HIGM, 411: CLASS_HIGM, 421: CLASS_HIGM, 565: CLASS_HIGM, 569: CLASS_HIGM,
    574: CLASS_HIGM, 342: CLASS_HIGM, 346: CLASS_HIGM, 326: CLASS_HIGM, 347: CLASS_HIGM, 576: CLASS_HIGM,
    558: CLASS_HIGM, 404: CLASS_HIGM, 364: CLASS_HIGM, 405: CLASS_HIGM, 559: CLASS_HIGM, 365: CLASS_HIGM,
    561: CLASS_HIGM, 407: CLASS_HIGM, 367: CLASS_HIGM, 575: CLASS_HIGM, 422: CLASS_HIGM, 432: CLASS_HIGM,
    320: CLASS_HIGM,
    # --- Hippocampal WM ---
    461: CLASS_HIWM, 343: CLASS_HIWM, 368: CLASS_HIWM, 372: CLASS_HIWM, 408: CLASS_HIWM, 418: CLASS_HIWM,
    562: CLASS_HIWM, 566: CLASS_HIWM, 571: CLASS_HIWM, 339: CLASS_HIWM, 354: CLASS_HIWM,
    # --- Amygdala ---
    238: CLASS_AM, 277: CLASS_AM, 278: CLASS_AM, 279: CLASS_AM, 301: CLASS_AM, 377: CLASS_AM, 217: CLASS_AM,
    240: CLASS_AM, 242: CLASS_AM, 214: CLASS_AM, 215: CLASS_AM, 216: CLASS_AM, 295: CLASS_AM,
    # --- Thalamus ---
    276: CLASS_TH, 444: CLASS_TH, 430: CLASS_TH, 201: CLASS_TH, 314: CLASS_TH, 381: CLASS_TH, 382: CLASS_TH,
    394: CLASS_TH, 458: CLASS_TH, 283: CLASS_TH, 493: CLASS_TH, 519: CLASS_TH, 191: CLASS_TH, 312: CLASS_TH,
    285: CLASS_TH, 425: CLASS_TH, 284: CLASS_TH, 396: CLASS_TH, 479: CLASS_TH, 350: CLASS_TH, 222: CLASS_TH,
    397: CLASS_TH, 218: CLASS_TH, 492: CLASS_TH, 322: CLASS_TH, 442: CLASS_TH, 190: CLASS_TH, 246: CLASS_TH,
    517: CLASS_TH, 478: CLASS_TH, 226: CLASS_TH, 303: CLASS_TH, 398: CLASS_TH, 426: CLASS_TH, 424: CLASS_TH,
    400: CLASS_TH, 274: CLASS_TH, 395: CLASS_TH, 423: CLASS_TH, 811: CLASS_TH, 441: CLASS_TH, 504: CLASS_TH,
    512: CLASS_TH, 510: CLASS_TH, 378: CLASS_TH, 221: CLASS_TH, 379: CLASS_TH, 313: CLASS_TH, 282: CLASS_TH,
    225: CLASS_TH, 224: CLASS_TH, 399: CLASS_TH, 286: CLASS_TH, 223: CLASS_TH, 454: CLASS_TH, 380: CLASS_TH,
    813: CLASS_TH, 219: CLASS_TH, 220: CLASS_TH, 252: CLASS_TH, 253: CLASS_TH, 443: CLASS_TH, 508: CLASS_TH,
    578: CLASS_TH,
    # --- Pallidum ---
    119: CLASS_PAL, 206: CLASS_PAL,
    # --- Putamen ---
    79: CLASS_PU, 349: CLASS_PU,
    # --- Caudate + Accumbens ---
    48: CLASS_CDAC, 118: CLASS_CDAC, 393: CLASS_CDAC, 101: CLASS_CDAC, 184: CLASS_CDAC,
    # --- Cerebellar white ---
    846: CLASS_CBWM,
    # --- Cerebellar gray (and cerebellar “other” folded to gray) ---
    595: CLASS_CBCX, 597: CLASS_CBCX, 721: CLASS_CBCX, 715: CLASS_CBCX, 751: CLASS_CBCX, 752: CLASS_CBCX,
    # --- Brainstem ---
    496: CLASS_BS, 498: CLASS_BS, 465: CLASS_BS, 114: CLASS_BS, 412: CLASS_BS, 414: CLASS_BS, 521: CLASS_BS,
    654: CLASS_BS, 666: CLASS_BS, 687: CLASS_BS, 765: CLASS_BS, 843: CLASS_BS, 435: CLASS_BS, 541: CLASS_BS,
    384: CLASS_BS, 451: CLASS_BS, 506: CLASS_BS, 580: CLASS_BS, 611: CLASS_BS, 697: CLASS_BS, 531: CLASS_BS,
    662: CLASS_BS,
    # --- Substantia Nigra / STN / RN (ventral DC) ---
    310: CLASS_DC, 352: CLASS_DC, 315: CLASS_DC, 316: CLASS_DC, 321: CLASS_DC, 385: CLASS_DC,
    # --- Hypothalamus gray (ventral DC) ---
    256: CLASS_DC, 275: CLASS_DC, 304: CLASS_DC, 147: CLASS_DC, 150: CLASS_DC, 192: CLASS_DC, 193: CLASS_DC,
    194: CLASS_DC, 207: CLASS_DC, 227: CLASS_DC, 228: CLASS_DC, 229: CLASS_DC, 243: CLASS_DC, 244: CLASS_DC,
    245: CLASS_DC, 255: CLASS_DC, 268: CLASS_DC, 297: CLASS_DC, 308: CLASS_DC, 149: CLASS_DC, 160: CLASS_DC,
    181: CLASS_DC, 196: CLASS_DC, 230: CLASS_DC, 232: CLASS_DC,
    # --- Hypothalamus white (ventral DC) ---
    305: CLASS_DC, 306: CLASS_DC, 307: CLASS_DC, 234: CLASS_DC, 298: CLASS_DC, 309: CLASS_DC, 433: CLASS_DC,
    # --- Other DC ---
    484: CLASS_WM,  # NB: dorsal lateral geniculate nucleus → folded to WM (as agreed)
}

def _nextbrain_remap_numpy(lab_np: np.ndarray) -> np.ndarray:
    """Vectorized remap using LUT when possible + cortex override."""
    if lab_np.size == 0:
        return lab_np.astype(np.int16, copy=False)

    lab_np = lab_np.astype(np.int64, copy=False)
    max_id = int(lab_np.max())

    # Fast LUT path (for common case)
    out = np.zeros_like(lab_np, dtype=np.int16)
    if max_id < 100000:
        lut = np.zeros(max_id + 1, dtype=np.int16)
        for src, dst in NEXT_TO_ASEG.items():
            if 0 <= src <= max_id:
                lut[src] = np.int16(dst)
        out = lut[lab_np]
    else:
        # Sparse loop fallback
        for src, dst in NEXT_TO_ASEG.items():
            out[lab_np == src] = np.int16(dst)

    # Cortex override
    cortex_mask = lab_np > _CORTEX_RULE_THRESHOLD
    if cortex_mask.any():
        out[cortex_mask] = np.int16(CLASS_CX)

    return out

class MapLabelsSynthSeg(MapTransform):
    """
    NextBrain → compact SynthSeg classes (BG..14). Mirrors offline mapping.
    If inputs are already compact (max <= 14), acts as a no-op passthrough.
    """
    @staticmethod
    def label_mapping():
        # Expose the compact class set (BG..14) to downstream config if needed.
        return list(range(15))  # 0..14 inclusive

    def __init__(self, key_label: str = "label"):
        self.key_label = key_label

    def __call__(self, data):
        d = dict(data)
        lab = d[self.key_label]
        meta = getattr(lab, "meta", None)
        lab_np = lab.as_tensor().cpu().numpy() if isinstance(lab, MetaTensor) else np.asarray(lab)

        # If already compact (e.g., remapped offline), keep as-is
        if lab_np.size and int(lab_np.max()) <= 14:
            out_np = lab_np.astype(np.int16, copy=False)
        else:
            out_np = _nextbrain_remap_numpy(lab_np)

        out = MetaTensor(torch.from_numpy(out_np)[None])  # add channel dim
        if meta is not None:
            out.copy_meta_from(meta)
        d[self.key_label] = out
        return d

def ResizeTransform(keys: KeysCollection, spatial_size: Union[Sequence[int], int], method: str):
    """ Returns a resize transform. """
    if method == "pad_crop":
        return ResizeWithPadOrCropd(keys=keys, spatial_size=spatial_size)
    elif method == "spatial":
        return Resized(keys=keys, spatial_size=spatial_size, mode="nearest")
    else:
        raise ValueError(f"Undefined resize transform {method}")

class SynthSegd(MapTransform, RandomizableTransform):
    """
    Cornucopia-based Synth-from-Labels transform to render synthetic MR images
    from labelmaps (optionally cropped to patches).
    """
    def __init__(self, key_image="image", key_label="label", patch_size=None, params=None):
        RandomizableTransform.__init__(self, 1.0)
        self.key_image = key_image
        self.key_label = key_label
        self.patch_size = patch_size
        self.params = {} if params is None else dict(params)

    def __call__(self, data):
        d = dict(data)

        label = d[self.key_label]
        meta = getattr(label, "meta", None)
        label = label.as_tensor() if isinstance(label, MetaTensor) else torch.as_tensor(label)

        synth = cc.SynthFromLabelTransform(**self.params, patch=self.patch_size)
        image, label = synth(label)

        # mid = [s // 2 for s in label.shape[-3:]]
        # slicer = (slice(m - ps // 2, m + (ps + 1) // 2) for m, ps in zip(mid, self.patch_size))
        # slicer = (...,) + tuple(slicer)
        # label = label[slicer]
        # image = label.clone().float()

        d[self.key_image] = MetaTensor(image)
        if meta is not None:
            d[self.key_image].copy_meta_from(meta)

        d[self.key_label] = MetaTensor(label)
        if meta is not None:
            d[self.key_label].copy_meta_from(meta)

        return d

class LabelUnlabelledGMM(MapTransform):
    """Optional: label unlabelled voxels via a GMM over intensities; keeps class IDs disjoint."""
    def __init__(self, n_labels=None, num_gmm_classes=10, key_label="label", key_image="image", verbose=False):
        self.n_labels = n_labels
        self.key_label = key_label
        self.key_image = key_image
        self.num_gmm_classes = num_gmm_classes
        self.verbose = verbose

    def __call__(self, data):
        d = dict(data)
        label = d[self.key_label]; image = d[self.key_image]
        meta = label.meta if isinstance(label, MetaTensor) else None
        label = label.as_tensor() if isinstance(label, MetaTensor) else torch.as_tensor(label)
        image = image.as_tensor() if isinstance(image, MetaTensor) else torch.as_tensor(image)

        lab = label[0]; img = image[0]
        if self.n_labels is None:
            self.n_labels = int(lab.max().item())
        unlabelled = lab == 0
        intens = img[unlabelled].reshape(1, -1).float()

        Zu = cc.utils.gmm.fit_gmm(intens, nk=self.num_gmm_classes, max_iter=1024)[0]
        Z = torch.zeros((self.num_gmm_classes, unlabelled.numel()), dtype=Zu.dtype, device=Zu.device)
        for k in range(self.num_gmm_classes):
            Z[k, unlabelled.flatten()] = Zu[k, :]
        Z = Z.reshape((self.num_gmm_classes,) + tuple(unlabelled.shape)).permute(1, 2, 3, 0)
        Z = torch.argmax(Z, dim=-1).type(lab.dtype)

        # merge new background (0) into 0; shift others above existing label IDs
        lab[(unlabelled) & (Z > 0)] = (Z[(unlabelled) & (Z > 0)] + self.n_labels)
        out = MetaTensor(lab[None])
        if meta is not None:
            out.copy_meta_from(meta)
        d[self.key_label] = out
        return d
