"""Code for custom transforms for Synthseg training in MONAI.
"""
import cornucopia as cc
import torch
from random import randrange
from monai.transforms import (
    MapTransform,
    RandomizableTransform,
    Resized,
    ResizeWithPadOrCropd,
)
from monai.data.meta_tensor import MetaTensor

# === NextBrain → SynthSeg remapper ===========================================
import numpy as _np
from monai.transforms import MapTransform as _MapTransform


CLASS_BG  = 0  # Background + CSF + ventricles
CLASS_WM  = 1  # Cerebral white matter
CLASS_GM  = 2  # Cerebral gray matter (not cortex)
CLASS_CX  = 3  # Cerebral cortex (gray matter)
CLASS_CBWM= 4  # Cerebellar white matter
CLASS_CBCX= 5  # Cerebellar cortex (gray matter)
CLASS_TH  = 6  # Thalamus
CLASS_CDAC= 7  # Caudate + Accumbens
CLASS_PU  = 8  # Putamen
CLASS_PAL = 9  # Pallidum
CLASS_BS  = 10 # Brain stem
CLASS_HIGM= 11 # Hippocampus WM
CLASS_HIWM= 12 # Hippocampus GM
CLASS_AM  = 13 # Amygdala
CLASS_DC  = 14 # Ventral DC

_CORTEX_RULE_THRESHOLD = 1000

NEXT_TO_ASEG = {
    # --- White matter ---
    7:   CLASS_WM, 68:  CLASS_WM, 120: CLASS_WM, 130: CLASS_WM, 199: CLASS_WM,
    100: CLASS_WM, 161: CLASS_WM, 208: CLASS_WM, 209: CLASS_WM, 102: CLASS_WM,
    174: CLASS_WM, 254: CLASS_WM,
    # --- Gray matter ---
    103: CLASS_GM, 108: CLASS_GM, 111: CLASS_GM, 117: CLASS_GM, 128: CLASS_GM,
    99: CLASS_GM,  113: CLASS_GM, 125: CLASS_GM, 157: CLASS_GM,
    # --- Hippocampal gray matter ---
    344: CLASS_HIGM, 369: CLASS_HIGM, 409: CLASS_HIGM, 567: CLASS_HIGM, 340: CLASS_HIGM, 563: CLASS_HIGM,
    419: CLASS_HIGM, 373: CLASS_HIGM, 572: CLASS_HIGM, 345: CLASS_HIGM, 370: CLASS_HIGM, 374: CLASS_HIGM,
    410: CLASS_HIGM, 420: CLASS_HIGM, 564: CLASS_HIGM, 573: CLASS_HIGM, 341: CLASS_HIGM, 568: CLASS_HIGM,
    371: CLASS_HIGM, 375: CLASS_HIGM, 411: CLASS_HIGM, 421: CLASS_HIGM, 565: CLASS_HIGM, 569: CLASS_HIGM,
    574: CLASS_HIGM, 342: CLASS_HIGM, 346: CLASS_HIGM, 326: CLASS_HIGM, 347: CLASS_HIGM, 576: CLASS_HIGM,
    558: CLASS_HIGM, 404: CLASS_HIGM, 364: CLASS_HIGM, 405: CLASS_HIGM, 559: CLASS_HIGM, 365: CLASS_HIGM,
    561: CLASS_HIGM, 407: CLASS_HIGM, 367: CLASS_HIGM, 575: CLASS_HIGM, 422: CLASS_HIGM, 432: CLASS_HIGM,
    320: CLASS_HIGM,
    # --- Hippocampal White Matter ---
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
    # --- Cerebellar White ---
    846: CLASS_CBWM,
    # --- Cerebellar Gray + other to cb gray ---
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
    484: CLASS_WM,
}

class RemapNextBrainLabelsd(_MapTransform):
    """
    Remap raw NextBrain labels to compact SynthSeg-style classes.
    Use this BEFORE any further label packing or synthesis steps.
    """
    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self._lut = None

    def _ensure_lut(self, max_id: int):
        if self._lut is None or len(self._lut) <= max_id:
            lut = _np.zeros(max_id + 1, dtype=_np.int16)
            for src, dst in NEXT_TO_ASEG.items():
                if 0 <= src <= max_id:
                    lut[src] = _np.int16(dst)
            self._lut = lut

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            arr = _np.asarray(d[k])
            if not _np.issubdtype(arr.dtype, _np.integer):
                arr = _np.rint(arr).astype(_np.int64, copy=False)
            else:
                arr = arr.astype(_np.int64, copy=False)

            flat = arr.reshape(-1)
            max_id = int(flat.max()) if flat.size else 0

            if max_id < 100000:
                self._ensure_lut(max_id)
                out_flat = self._lut[flat]
            else:
                out_flat = _np.zeros_like(flat, dtype=_np.int16)
                for src, dst in NEXT_TO_ASEG.items():
                    m = (flat == src)
                    if m.any():
                        out_flat[m] = _np.int16(dst)

            # Cortex override: any label > 1000 is cortex
            if max_id > _CORTEX_RULE_THRESHOLD:
                m_cx = (flat > _CORTEX_RULE_THRESHOLD)
                if m_cx.any():
                    out_flat[m_cx] = _np.int16(CLASS_CX)

            d[k] = out_flat.reshape(arr.shape).astype(_np.int16, copy=False)
        return d

class LabelUnlabelledGMM(RandomizableTransform):
    """(Existing class from your file)
    Finds unlabeled voxels, fits a small GMM on their intensities, and assigns them
    temporary labels > n_labels to increase class diversity.
    """
    def __init__(self, key_image: str = "image", key_label: str = "label", n_gmm_classes: int = 3, n_labels: int = 1, verbose: bool = False):
        super().__init__()
        self.key_image = key_image
        self.key_label = key_label
        self.n_gmm_classes = n_gmm_classes
        self.n_labels = n_labels
        self.verbose = verbose

    def __call__(self, d):
        import torch
        image: MetaTensor = d[self.key_image]
        label: MetaTensor = d[self.key_label]
        X = image.squeeze().flatten()
        Y = label.squeeze().flatten()
        unlabelled = (Y == 0)
        X_u = X[unlabelled].float()
        if X_u.numel() == 0:
            return d
        # Simple k-means-ish 1D clustering as stand-in for tiny GMM
        # (kept lightweight; assumes normalized intensities)
        k = max(2, self.n_gmm_classes)
        # Random init
        cents = torch.linspace(X_u.min(), X_u.max(), steps=k)
        for _ in range(3):
            dists = (X_u[:, None] - cents[None, :]).abs()
            Z = dists.argmin(dim=1)
            for i in range(k):
                sel = (Z == i)
                if sel.any():
                    cents[i] = X_u[sel].mean()
        Z = dists.argmin(dim=1)
        if self.verbose:
            print(f"LabelUnlabelledGMM | Number of output GMM classes: {len(Z.unique())}")
        # Correct data type
        Z = Z.type(label.dtype)
        # Assign new labels
        label[(unlabelled) & (Z == 0)] = Z[(unlabelled) & (Z == 0)]
        label[(unlabelled) & (Z > 0)] = (Z[(unlabelled) & (Z > 0)] + self.n_labels)
        if self.verbose:
            print(f"LabelUnlabelledGMM | Number of unique output labels (inc zero): {len(label.unique())}")
        d[self.key_label] = MetaTensor(label[None])
        return d


class MapLabelsSynthSeg(MapTransform):
    """(Existing class from your file)
    Map (ASEG-like) labels to contiguous indices starting at one.
    """
    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    @staticmethod
    def label_mapping():
        return {
            2:1,
            3:2,
            4:3,
            5:4,
            7:5,
            8:6,
            10:7,
            11:8,
            12:9,
            13:10,
            14:11,
            15:12,
            16:13,
            17:14,
            18:15,
            24:16,
            28:17,
            41:18,
            42:19,
            43:20,
            44:21,
            46:22,
            47:23,
            49:24,
            50:25,
            51:26,
            52:27,
            53:28,
            54:29,
            58:30,
            60:31,
        }

    def __call__(self, d):
        import torch
        mapping = self.label_mapping()
        for k in self.keys:
            lbl: MetaTensor = d[k]
            x = lbl.as_tensor().long()
            out = torch.zeros_like(x)
            for src, dst in mapping.items():
                out[x == src] = dst
            d[k] = MetaTensor(out)
        return d


class SynthSegd(MapTransform):
    """(Existing class from your file)
    Wrap Cornucopia's SynthFromLabelTransform with MONAI dict interface.
    """
    def __init__(self, key_label: str = "label", key_out: str = "image", patch_size=(96, 96, 96), **params):
        super().__init__([key_label])
        self.key_label = key_label
        self.key_out = key_out
        self.patch_size = patch_size
        self.params = params

    def __call__(self, d):
        label: MetaTensor = d[self.key_label]
        synth = cc.SynthFromLabelTransform(
            **self.params,
            patch=self.patch_size,
        )
        out = synth(label.squeeze())  # returns torch.Tensor [C, ...]
        d[self.key_out] = MetaTensor(out)
        return d

