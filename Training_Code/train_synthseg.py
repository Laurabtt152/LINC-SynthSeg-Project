#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import os
import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureTyped,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    RandFlipd,
    ToTensord,
    Lambdad,
)
from monai.data import (
    CacheDataset,
    DataLoader,
    decollate_batch,
    partition_dataset,
    Dataset,  # used for debug dry-run (no cache/threads)
)
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

import transforms_synthseg as transforms
import utils_synthseg as utils

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
seed = 0
set_determinism(seed=seed)
torch.backends.cudnn.benchmark = True

# Device (CUDA if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device)
print(f"[info] device = {device}")

# Paths & run config
# (Point to your repo's labels folder; this matches your working CPU script layout.)
dir_input = str((Path(__file__).resolve().parent / "labels").as_posix())
dir_results = "./results"
batch_size = 1
total_steps = 400000                 # total_steps ~= epochs * len(train_files)
validation_steps = 2000              # steps between validations
spatial_size = None                  # e.g., (256,)*3 or None
patch_size = (96,96,96)                   # e.g., (128,)*3 or None
pth_checkpoint = os.path.join(dir_results, "checkpoint.pkl")
pth_checkpoint_prev = os.path.join(dir_results, "checkpoint_prev.pkl")
os.makedirs(dir_results, exist_ok=True)

# ---------------------------------------------------------------------
# DDP (optional)
# ---------------------------------------------------------------------
if "LOCAL_RANK" in os.environ:
    print("Setting up DDP...", end="")
    ddp = True
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    num_gpus = dist.get_world_size()
    print("done!")
else:
    ddp = False
    num_gpus = 1

# ---------------------------------------------------------------------
# Collect label files
# ---------------------------------------------------------------------
print(f"[debug] dir_input = {dir_input}")
labels = [str(d) for d in sorted(Path(dir_input).rglob("*.nii.gz"))]
if len(labels) == 0:
    raise FileNotFoundError(
        f"No .nii.gz label maps found in '{dir_input}'. "
        f"Please populate this folder (or adjust dir_input)."
    )
data = [{"label": p} for p in labels]

# Same files for train/val (you mentioned val uses minimal randomness)
train_files, val_files = data, data

# DDP: partition per device
if ddp:
    train_files = partition_dataset(
        data=train_files,
        num_partitions=num_gpus,
        shuffle=False,
        seed=seed,
        drop_last=False,
        even_divisible=True,
    )[dist.get_rank()]

num_train = len(train_files)
num_val = len(val_files)
print(f"[debug] Dataset sizes — train={num_train}, val={num_val}")

# Steps/epochs (guard div-by-zero)
steps_per_epoch = max(1, num_train)
epochs = max(1, total_steps // steps_per_epoch)
validation_epoch = max(1, round(validation_steps / steps_per_epoch))
print(f"[debug] steps_per_epoch={steps_per_epoch}, epochs={epochs}, validation_epoch={validation_epoch}")

# ---------------------------------------------------------------------
# Label mapping / class info
# ---------------------------------------------------------------------
# Derive class ids from your MapLabelsSynthSeg mapping (exclude background 0).
# NOTE: this assumes your transforms.MapLabelsSynthSeg exposes label_mapping().
target_labels = sorted({v for v in transforms.MapLabelsSynthSeg.label_mapping().values() if v != 0})
n_labels = len(target_labels)       # foreground classes
out_channels = n_labels + 1         # + background
print(f"[debug] n_labels (FG)={n_labels}, out_channels (BG+FG)={out_channels}")

# ---------------------------------------------------------------------
# Transforms (split into pre-SynthSegd and SynthSegd-only for clearer debugging)
# ---------------------------------------------------------------------
def build_pre_tfm(train: bool):
    # everything up to (but NOT including) SynthSegd
    return Compose([
        LoadImaged(keys="label"),
        EnsureChannelFirstd(keys="label"),
        Orientationd(keys="label", axcodes="RAS"),
        transforms.MapLabelsSynthSeg(),
        transforms.ResizeTransform(keys=["label"], spatial_size=spatial_size, method="pad_crop"),
        # force integer labels (SynthSegd expects discrete classes)
        Lambdad(keys=("label",), func=lambda x: x.astype("int16", copy=False)),
        EnsureTyped(keys="label", dtype=torch.int16, device="cpu"),
    ])

def build_synth_only(train: bool):
    return transforms.SynthSegd(
        params=utils.get_synth_params(
            target_labels=list(range(1, out_channels)),  # foreground classes 1..C
            train=train
        ),
        patch_size=patch_size if train else None
    )

pre_train = build_pre_tfm(train=True)
pre_val   = build_pre_tfm(train=False)
syn_train = build_synth_only(train=True)
syn_val   = build_synth_only(train=False)

# final transforms used by loaders
train_transforms = Compose([pre_train, syn_train, ToTensord(keys=["image", "label"])])
val_transforms   = Compose([pre_val,   syn_val,   ToTensord(keys=["image", "label"])])

# ---------------------------------------------------------------------
# **DEBUG**: two-stage dry-run to expose the real SynthSegd error
# ---------------------------------------------------------------------
import traceback, json
os.environ["MONAI_DATA_THREAD"] = "0"  # disable threaded execution for clean tracebacks

def _desc(arr, name):
    arr = np.asarray(arr)
    msg = {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": int(arr.min()) if arr.size else None,
        "max": int(arr.max()) if arr.size else None,
    }
    if arr.size:
        uniq, cnt = np.unique(arr, return_counts=True)
        k = min(12, len(uniq))
        msg["uniq_sample"] = {int(u): int(c) for u, c in zip(uniq[:k], cnt[:k])}
    print("[debug] " + json.dumps(msg))

def debug_dry_run_one(sample_path: str, pre_tfm, syn_tfm, tag: str):
    print(f"[debug:{tag}] file={sample_path}")
    data = {"label": sample_path}
    # Stage 1: pre-SynthSegd
    try:
        data = pre_tfm(data)
    except Exception:
        print("\n[ERROR] Failed in pre-SynthSegd transforms.")
        traceback.print_exc()
        raise
    lab = data["label"]
    _desc(lab, f"{tag}/pre_synth_label")
    # Save pre-synth label to inspect if needed
    try:
        np.save(os.path.join(dir_results, f"debug_label_{tag}.npy"), lab)
        print(f"[debug:{tag}] saved pre-synth label to {dir_results}/debug_label_{tag}.npy")
    except Exception:
        pass

    # Stage 2: SynthSegd only
    try:
        out = syn_tfm(data.copy())
        img2 = out["image"]; lab2 = out["label"]
        _desc(img2, f"{tag}/post_synth_image")
        _desc(lab2, f"{tag}/post_synth_label")
    except Exception as e:
        print("\n[ERROR] SynthSegd raised an exception.")
        cause = e.__cause__
        if cause is not None:
            print("[cause type] ", type(cause))
            print("[cause str ] ", str(cause))
            print("[cause tb  ]")
            traceback.print_exception(type(cause), cause, cause.__traceback__)
        print("[wrapped traceback]")
        traceback.print_exc()
        raise

# pick one file and run the debug
nii_paths = sorted(glob(os.path.join(dir_input, "*.nii"))) + \
            sorted(glob(os.path.join(dir_input, "*.nii.gz")))
assert nii_paths, f"No NIfTI label maps found in {dir_input}"
debug_dry_run_one(nii_paths[0], pre_train, syn_train, "train")
debug_dry_run_one(nii_paths[0], pre_val,   syn_val,   "val")

# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
train_loader = DataLoader(
    CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0),
    batch_size=batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0),
    batch_size=1,
    shuffle=False,
)
print(f"[debug] train_loader batches={len(train_loader)} | val_loader batches={len(val_loader)}")
if len(train_loader) == 0:
    raise RuntimeError("No training batches; check labels folder, transforms, and patch_size.")

# ---------------------------------------------------------------------
# Model / loss / optimiser / metrics
# ---------------------------------------------------------------------
model = utils.get_model(out_channels).to(device)

loss_function = DiceLoss(
    to_onehot_y=True, softmax=True, include_background=True,
    smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=True,
)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

optimizer = torch.optim.Adam(model.parameters(), lr=math.sqrt(batch_size * num_gpus) * 1e-4)
scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

# ---------------------------------------------------------------------
# Load checkpoint (robust)
# ---------------------------------------------------------------------
if os.path.isfile(pth_checkpoint):
    print(f"[info] Loading checkpoint from {pth_checkpoint}")
    checkpoint = torch.load(pth_checkpoint, map_location=device)
    state = checkpoint.get("model_state_dict", {})
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(f"[warn] could not load optimizer state: {e}")
    epoch = int(checkpoint.get("epoch", 0))
    best_metric = float(checkpoint.get("best_metric", -1))
    loss_values = list(checkpoint.get("loss_values", []))
    metric_values = list(checkpoint.get("metric_values", []))
    train_duration = float(checkpoint.get("train_duration", 0))
else:
    checkpoint = {}
    epoch = 0
    best_metric = -1.0
    loss_values, metric_values = [], []
    train_duration = 0.0

# DDP wrap (if applicable)
if ddp:
    model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)

post_pred = Compose([Activations(softmax=True), AsDiscrete(threshold=0.5)])
fig, axs = plt.subplots(4, 3, figsize=(9, 12))
cmap, norm = utils.get_label_cmap(n_labels=out_channels)

print("-" * 64)
print(f"Starting training from epoch {epoch}/{epochs}")
print("-" * 64)

# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
for epoch in range(epoch, epochs):

    epoch_loss = 0.0
    start_time = time.time()

    # rotate previous checkpoint
    if epoch > 0 and not math.isnan(epoch_loss):
        ckpt_prev = dict(
            epoch=epoch,
            best_metric=best_metric,
            loss_values=loss_values,
            metric_values=metric_values,
            train_duration=train_duration,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
        )
        try:
            torch.save(ckpt_prev, pth_checkpoint_prev)
        except Exception as e:
            print(f"[warn] Error saving previous checkpoint: {e}")

    # -------------------------
    # Train
    # -------------------------
    model.train()
    step_epoch = 0
    save_fig_ix = np.random.randint(0, max(1, len(train_loader)))  # for debug figure

    for ix, batch_data in enumerate(train_loader):
        image = batch_data["image"].to(device)
        label = batch_data["label"].to(device)

        step_epoch += 1
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda" and scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(image)
                loss = loss_function(pred, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(image)
            loss = loss_function(pred, label)
            loss.backward()
            optimizer.step()

        epoch_loss += float(loss.item())

        if (epoch % validation_epoch == 0 or epoch == epochs - 1) and (save_fig_ix == ix):
            with torch.no_grad():
                pred_list = [post_pred(i) for i in decollate_batch(pred)]
            bx = 0
            axs[0, 0].imshow(utils.extract_slice(image[bx][0]).detach().cpu().numpy(), cmap="gray")
            axs[0, 0].set_title("train image"); axs[0, 0].axis("off")
            axs[0, 1].imshow(utils.extract_slice(label[bx][0]).detach().cpu().numpy(),
                             cmap=cmap, norm=norm, interpolation="nearest")
            axs[0, 1].set_title("train label"); axs[0, 1].axis("off")
            axs[0, 2].imshow(utils.extract_slice(pred_list[bx].argmax(0)).detach().cpu().numpy(),
                             cmap=cmap, norm=norm, interpolation="nearest")
            axs[0, 2].set_title("train pred"); axs[0, 2].axis("off")

    epoch_loss /= max(1, step_epoch)

    print(f"EPOCH={epoch + 1:{' '}{len(str(epochs))}}/{epochs} |__LOSS (N_t={num_train}, N_b={batch_size})={epoch_loss:.4f}")
    loss_values.append(epoch_loss)

    # -------------------------
    # Validate
    # -------------------------
    if epoch % validation_epoch == 0 or epoch == epochs - 1:
        model.eval()
        save_fig_ix = np.random.randint(0, max(1, len(val_loader)))
        with torch.no_grad():
            with torch.random.fork_rng(enabled=seed):
                torch.random.manual_seed(seed)
                for ix, batch_data in enumerate(val_loader):
                    image = batch_data["image"].to(device)
                    label = batch_data["label"].to(device)

                    pred = utils.inference(image, model, patch_size=patch_size)
                    pred_list = [post_pred(i) for i in decollate_batch(pred)]

                    dice_metric(y_pred=pred_list, y=label)
                    dice_metric_batch(y_pred=pred_list, y=label)

                    if ix == 0:
                        bx = 0
                        inputs_slice = utils.extract_slice(image[bx][0]).detach().cpu().numpy()
                        labels_slice = utils.extract_slice(label[bx][0]).detach().cpu().numpy()
                        pred_slice = utils.extract_slice(pred_list[bx].argmax(0)).detach().cpu().numpy()
                        axs[1, 0].imshow(inputs_slice, cmap="gray")
                        axs[1, 0].set_title("val image now"); axs[1, 0].axis("off")
                        axs[1, 1].imshow(labels_slice, cmap=cmap, norm=norm, interpolation="nearest")
                        axs[1, 1].set_title("val label now"); axs[1, 1].axis("off")
                        axs[1, 2].imshow(pred_slice, cmap=cmap, norm=norm, interpolation="nearest")
                        axs[1, 2].set_title("val pred now"); axs[1, 2].axis("off")

            # aggregate dice
            metric = dice_metric.aggregate().item()
            metric_batch = dice_metric_batch.aggregate()
            metric_values.append(metric)
            dice_metric.reset()
            dice_metric_batch.reset()

            print(f"EPOCH={epoch + 1:{' '}{len(str(epochs))}}/{epochs} |__METRIC (N_v={num_val}, N_b={batch_size})={metric:.4f}")
            per_line = 10
            for i in range(0, len(metric_batch), per_line):
                line = ",".join([f"{k+i:3d}={float(v):0.3f}" for k, v in enumerate(metric_batch[i:i+per_line])])
                print(" " * (13 + len(str(epochs))) + "|____" + line)

            # best snapshot
            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), os.path.join(dir_results, "model_best.pth"))
                axs[2, 0].imshow(inputs_slice, cmap="gray")
                axs[2, 0].set_title("val image best"); axs[2, 0].axis("off")
                axs[2, 1].imshow(labels_slice, cmap=cmap, norm=norm, interpolation="nearest")
                axs[2, 1].set_title("val label best"); axs[2, 1].axis("off")
                axs[2, 2].imshow(pred_slice, cmap=cmap, norm=norm, interpolation="nearest")
                axs[2, 2].set_title("val pred best"); axs[2, 2].axis("off")

            utils.plot_loss_and_metric(axs, loss_values, metric_values, validation_epoch)
            plt.suptitle(f"EPOCH={epoch}, LOSS={epoch_loss:.4f}, METRIC_NOW={metric:.4f}, METRIC_BEST={best_metric:.4f}")
            fig.tight_layout()
            plt.savefig(os.path.join(dir_results, "snapshot.png"))

    # time & checkpoint
    epoch_time = time.time() - start_time
    train_duration += epoch_time

    if math.isnan(epoch_loss):
        print("[warn] Loss is NaN; skipping checkpoint save")
    else:
        checkpoint = dict(
            epoch=epoch + 1,  # next epoch to start from
            best_metric=best_metric,
            loss_values=loss_values,
            metric_values=metric_values,
            train_duration=train_duration,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
        )
        try:
            torch.save(checkpoint, pth_checkpoint)
        except Exception as e:
            print(f"[warn] Error saving model: {e}")
