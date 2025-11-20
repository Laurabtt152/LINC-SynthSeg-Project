import math, os, time
from pathlib import Path
import nvidia_smi
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import matplotlib.pyplot as plt

from monai.transforms import (
    Activations, AsDiscrete, Compose, EnsureTyped, EnsureChannelFirstd,
    LoadImaged, Orientationd, ScaleIntensityd, RandFlipd,
)
from monai.data import CacheDataset, DataLoader, decollate_batch, partition_dataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

import transforms_synthseg as transforms
import utils_synthseg as utils

# -------------------------
# Repro + device
# -------------------------
seed = 0
set_determinism(seed=seed)
torch.backends.cudnn.benchmark = True
nvidia_smi.nvmlInit()
gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

# -------------------------
# Paths & settings
# -------------------------
dir_input = "./training_labels"   # offline-remapped NextBrain labels (.nii.gz)
dir_results = "./results"
batch_size = 1
total_steps = 400000
validation_steps = 50
spatial_size = None  # e.g., (256,)*3; None → no resize
patch_size = (128,) * 3    # e.g., (128,)*3; None → full volume
os.makedirs(dir_results, exist_ok=True)
pth_checkpoint      = os.path.join(dir_results, 'checkpoint.pkl')
pth_checkpoint_prev = os.path.join(dir_results, 'checkpoint_prev.pkl')
device = torch.device("cuda:0")

# -------------------------
# DDP setup
# -------------------------
if device != torch.device("cpu"):
    if "LOCAL_RANK" in os.environ:
        ddp = True
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        num_gpus = dist.get_world_size()
    else:
        ddp = False
        num_gpus = 1
    torch.cuda.set_device(device)
else:
    ddp = False
    num_gpus = 1

# -------------------------
# Data listing
# -------------------------
labels = [str(p) for p in sorted(Path(dir_input).rglob("*.nii.gz"))]
data = [{"label": p} for p in labels]
train_files, val_files = data, data  # same set; val uses near-deterministic synth params
num_train, num_val = len(train_files), len(val_files)
steps_per_epoch = max(1, num_train)
validation_epoch = max(1, round(validation_steps / steps_per_epoch))
if ddp:
    train_files = partition_dataset(
        data=train_files, num_partitions=num_gpus, shuffle=False,
        seed=seed, drop_last=False, even_divisible=True
    )[dist.get_rank()]

# -------------------------
# Unified class space (BG..14)
# -------------------------
target_labels = list(range(1, 15))  # 1..14 inclusive (do not include BG=0 !!!)
n_labels = len(target_labels)       # foreground classes (without BG) for color convenience if needed

# -------------------------
# Transforms
# -------------------------
synth_params = utils.get_synth_params(target_labels, train=True)
synth_params["device"] = device
train_transforms = Compose([
    LoadImaged(keys="label"),
    EnsureChannelFirstd(keys="label"),
    Orientationd(keys="label", axcodes="RAS"),
    # transforms.MapLabelsSynthSeg(key_label="label"),  # <-- NextBrain mapping (no-op if already remapped)
    transforms.ResizeTransform(keys=["label"], spatial_size=spatial_size, method="pad_crop"),
    EnsureTyped(keys="label", dtype=torch.int16, device="cpu"),
    transforms.SynthSegd(params=synth_params, patch_size=patch_size),
    ScaleIntensityd(keys="image"),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
])

synth_params = utils.get_synth_params(target_labels, train=False)
synth_params["device"] = device
val_transforms = Compose([
    LoadImaged(keys="label"),
    EnsureChannelFirstd(keys="label"),
    Orientationd(keys="label", axcodes="RAS"),
    # transforms.MapLabelsSynthSeg(key_label="label"),
    transforms.ResizeTransform(keys=["label"], spatial_size=spatial_size, method="pad_crop"),
    EnsureTyped(keys="label", dtype=torch.int16, device="cpu"),
    transforms.SynthSegd(params=synth_params, patch_size=patch_size),
    ScaleIntensityd(keys="image"),
])

# -------------------------
# Dataloaders
# -------------------------
train_loader = DataLoader(
    CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.0),
    batch_size=batch_size, shuffle=True,
)
val_loader = DataLoader(
    CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0),
    batch_size=1, shuffle=False,
)
epochs = max(1, total_steps // max(1, num_train))

# -------------------------
# Model / loss / optim
# -------------------------
out_channels = n_labels + 1  # BG + 14 foreground
model = utils.get_model(out_channels).to(device)
loss_function = DiceLoss(
    to_onehot_y=True, softmax=True, include_background=True,
    smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=True,
)
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
optimizer = torch.optim.Adam(model.parameters(), lr=math.sqrt(batch_size * num_gpus) * 1e-4)

# -------------------------
# Checkpoint (load or init)
# -------------------------
if os.path.isfile(pth_checkpoint):
    print(f"Loading checkpoint from {pth_checkpoint}")
    checkpoint = torch.load(pth_checkpoint, map_location="cpu")
    checkpoint["model_state_dict"] = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", -1)
    loss_values = checkpoint.get("loss_values", [])
    metric_values = checkpoint.get("metric_values", [])
    train_duration = checkpoint.get("train_duration", 0)
else:
    epoch = 0
    best_metric = -1
    loss_values, metric_values = [], []
    train_duration = 0

# -------------------------
# DDP wrap (optional)
# -------------------------
if ddp:
    model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)

post_pred = Compose([Activations(softmax=True), AsDiscrete(threshold=0.5)])
fig, axs = plt.subplots(4, 3, figsize=(9, 12))
cmap, norm = utils.get_label_cmap(n_labels=out_channels)
scaler = torch.GradScaler()

# -------------------------
# Training loop
# -------------------------
print("-" * 64)
print(f"Starting training from epoch {epoch}")
print("-" * 64)

for epoch in range(epoch, epochs):
    start_time = time.time()
    model.train()
    epoch_loss = 0.0
    save_fig_ix = np.random.randint(0, len(train_loader)) if len(train_loader) else 0

    if epoch == 0 or math.isnan(epoch_loss):
        print("WARNING: Loss is NaN do not save previous checkpoint")
    else:
        prev = dict(
            epoch=epoch, best_metric=best_metric,
            loss_values=loss_values, metric_values=metric_values,
            train_duration=train_duration,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
        )
        try:
            torch.save(prev, pth_checkpoint_prev)
        except Exception as e:
            print(f"Error saving previous model: {e}")

    step_epoch = 0
    for ix, batch in enumerate(train_loader):
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(image)
            loss = loss_function(pred, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += float(loss.item())
        step_epoch += 1

        if (epoch % validation_epoch == 0 or epoch == epochs - 1) and (save_fig_ix == ix):
            with torch.no_grad():
                pred_post = [post_pred(i) for i in decollate_batch(pred)]
            axs[0, 0].imshow(utils.extract_slice(image[0][0]).cpu().numpy(), cmap="gray"); axs[0, 0].set_title('train image'); axs[0, 0].axis('off')
            axs[0, 1].imshow(utils.extract_slice(label[0][0]).cpu().numpy(), cmap=cmap, norm=norm, interpolation="nearest"); axs[0, 1].set_title('train label'); axs[0, 1].axis('off')
            axs[0, 2].imshow(utils.extract_slice(pred_post[0].argmax(0)).cpu().numpy(), cmap=cmap, norm=norm, interpolation="nearest"); axs[0, 2].set_title('train pred'); axs[0, 2].axis('off')

    epoch_loss = epoch_loss / max(1, step_epoch)
    gpu_info = nvidia_smi.nvmlDeviceGetMemoryInfo(gpu_handle)
    print(f"EPOCH={epoch + 1:{' '}{len(str(epochs))}}/{epochs} |__LOSS={epoch_loss:.4f} | VRAM: {gpu_info.used/1e6:.0f}/{gpu_info.total/1e6:.0f}MB", end="")
    elapsed = time.time() - start_time
    train_duration += elapsed
    print(f" | train_duration: {train_duration/3600:0.1f} h")
    loss_values.append(epoch_loss)

    # -------------------------
    # Validation
    # -------------------------
    if epoch % validation_epoch == 0 or epoch == epochs - 1:
        model.eval()
        save_fig_ix = np.random.randint(0, len(val_loader)) if len(val_loader) else 0
        with torch.no_grad():
            with torch.random.fork_rng(enabled=seed):
                torch.random.manual_seed(seed)
                for ix, batch in enumerate(val_loader):
                    image = batch["image"].to(device)
                    label = batch["label"].to(device)

                    pred = utils.inference(image, model, patch_size=patch_size)
                    pred = [post_pred(i) for i in decollate_batch(pred)]

                    dice_metric(y_pred=pred, y=label)
                    dice_metric_batch(y_pred=pred, y=label)

                    if ix == 0:
                        inputs_slice = utils.extract_slice(image[0][0]).cpu().numpy()
                        labels_slice = utils.extract_slice(label[0][0]).cpu().numpy()
                        pred_slice = utils.extract_slice(pred[0].argmax(0)).cpu().numpy()
                        axs[1, 0].imshow(inputs_slice, cmap="gray"); axs[1, 0].set_title('val image now'); axs[1, 0].axis('off')
                        axs[1, 1].imshow(labels_slice, cmap=cmap, norm=norm, interpolation="nearest"); axs[1, 1].set_title('val label now'); axs[1, 1].axis('off')
                        axs[1, 2].imshow(pred_slice, cmap=cmap, norm=norm, interpolation="nearest"); axs[1, 2].set_title('val pred now'); axs[1, 2].axis('off')

            metric = dice_metric.aggregate().item()
            metric_batch = dice_metric_batch.aggregate()
            metric_values.append(metric)
            dice_metric.reset(); dice_metric_batch.reset()

            print(f"EPOCH={epoch + 1:{' '}{len(str(epochs))}}/{epochs} |__METRIC={metric:.4f}")
            for i in range(0, len(metric_batch), 10):
                vals = ",".join([f"{k + i:3.0f}={v:0.3f}" for k, v in enumerate(metric_batch[i:i + 10])])
                print(" " * (13 + len(str(epochs))) + "|____" + vals)

            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), os.path.join(dir_results, "model_best.pth"))
                axs[2, 0].imshow(inputs_slice, cmap="gray"); axs[2, 0].set_title('val image best'); axs[2, 0].axis('off')
                axs[2, 1].imshow(labels_slice, cmap=cmap, norm=norm, interpolation="nearest"); axs[2, 1].set_title('val label best'); axs[2, 1].axis('off')
                axs[2, 2].imshow(pred_slice, cmap=cmap, norm=norm, interpolation="nearest"); axs[2, 2].set_title('val pred best'); axs[2, 2].axis('off')

            utils.plot_loss_and_metric(axs, loss_values, metric_values, validation_epoch)
            plt.suptitle(f"EPOCH={epoch}, LOSS={epoch_loss:.4f}, METRIC_NOW={metric:.4f}, METRIC_BEST={best_metric:.4f}")
            fig.tight_layout()
            plt.savefig(os.path.join(dir_results, "snapshot.png"))

    # -------------------------
    # Save checkpoint
    # -------------------------
    if not math.isnan(epoch_loss):
        checkpoint = dict(
            epoch=epoch, best_metric=best_metric,
            loss_values=loss_values, metric_values=metric_values,
            train_duration=train_duration,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
        )
        try:
            torch.save(checkpoint, pth_checkpoint)
        except Exception as e:
            print(f"Error saving model: {e}")
