"""
Utility methods for Refrakt including patch operations, dataset utilities,
visualization, positional embeddings, and masking strategies.
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from torch import nn
from torchvision import transforms  # type: ignore

matplotlib.use("Agg")


NOISE_FACTOR = 0.1
SCALE_FACTOR = 4


def patchify(images: torch.Tensor, n_patches: int) -> torch.Tensor:
    """Convert images into non-overlapping patches."""
    n, c, h, w = images.shape
    assert h == w, "Only square images supported"
    patch_size = h // n_patches
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    patches = unfold(images).transpose(1, 2)
    return cast(torch.Tensor, patches.reshape(n, n_patches**2, -1))


def positional_embeddings(sequence_length: int, d: int) -> torch.Tensor:
    """Generate sinusoidal positional embeddings."""
    position = torch.arange(sequence_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2) * -(np.log(10000.0) / d))
    embeddings = torch.zeros(sequence_length, d)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings


def visualize_reconstructions(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader[Any],
    ae_type: str,
    device: torch.device,
    num_samples: int = 5,
) -> None:
    """Visualize original and reconstructed images from an autoencoder."""
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device).view(data.size(0), -1)

        reconstructed = None
        if ae_type in ("simple", "regularized"):
            reconstructed = model(data)
        elif ae_type == "denoising":
            noisy_data = (
                data * (1 - NOISE_FACTOR) + torch.rand_like(data) * NOISE_FACTOR
            )
            reconstructed = model(noisy_data)
        elif ae_type == "vae":
            _, reconstructed, _, _ = model(data)

        if reconstructed is None:
            raise ValueError("Autoencoder type not supported or reconstruction failed.")

        data = data.cpu().view(-1, 28, 28)
        reconstructed = reconstructed.cpu().view(-1, 28, 28)

        plt.figure(figsize=(10, 4))
        for i in range(num_samples):
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(data[i], cmap="gray")
            plt.title("Original")
            plt.axis("off")

            plt.subplot(2, num_samples, num_samples + i + 1)
            plt.imshow(reconstructed[i], cmap="gray")
            plt.title("Reconstructed")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{ae_type}_autoencoder_reconstructions.png")
        plt.close()


def download_dataset(root_dir: Path) -> None:
    """Download and unzip the dataset into the given root directory."""
    if root_dir.exists():
        print("Directory already exists.")
        return

    print("Creating the directory...")
    root_dir.mkdir(parents=True, exist_ok=True)

    dataset_zip = root_dir / "dataset.zip"
    with open(dataset_zip, "wb") as file:
        response = requests.get(
            "https://figshare.com/ndownloader/files/38256855", timeout=60
        )
        print("Downloading...")
        file.write(response.content)

    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        print("Unzipping...")
        zip_ref.extractall(root_dir)


def create_dirs(root_dir: Path) -> None:
    """Create dataset subdirectories for low-res and high-res images."""
    dataset_dir = root_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    (dataset_dir / "lr").mkdir(exist_ok=True)
    (dataset_dir / "hr").mkdir(exist_ok=True)


def count_folders(root_dir: Path) -> Tuple[int, int]:
    """Recursively count folders and files under a directory."""
    folder_count, file_count = 0, 0
    for folder in root_dir.iterdir():
        if folder.is_dir():
            sub_folders, sub_files = count_folders(folder)
            folder_count += 1 + sub_folders
            file_count += sub_files
        else:
            file_count += 1
    return folder_count, file_count


def move_images(lr_dir: Path, hr_dir: Path, root_dir: Path) -> None:
    """Move *_LR.png and *_HR.png files to designated directories."""
    for folder in root_dir.iterdir():
        if folder.is_dir():
            for file in folder.iterdir():
                if file.name.endswith("_LR.png"):
                    file.rename(lr_dir / file.name)
                elif file.name.endswith("_HR.png"):
                    file.rename(hr_dir / file.name)


def delete_dir(target_dir: Path) -> None:
    """Delete a directory recursively."""
    try:
        shutil.rmtree(target_dir)
        print(f"Removed directory: {target_dir}")
    except OSError as e:
        print(f"Error: {e.strerror}")


def get_transform(
    is_hr: bool = True, scale_factor: int = SCALE_FACTOR
) -> transforms.Compose:
    """Return transform pipeline for HR or LR images."""
    size = (32 * scale_factor, 32 * scale_factor) if is_hr else (32, 32)
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Return class names and mapping from directory structure."""
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't load any classes in {directory}")
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> torch.Tensor:
    """Generate 2D sine-cosine positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        cls_pos = np.zeros([1, embed_dim])
        pos_embed = np.concatenate([cls_pos, pos_embed], axis=0)
    return torch.tensor(pos_embed, dtype=torch.float32)


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed(
    embed_dim: int, pos: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    pos = pos.reshape(-1)
    out = np.outer(pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def random_masking(
    x: torch.Tensor, mask_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply per-sample random masking.
    Returns masked tensor, mask, ids_restore, and ids_keep."""
    b, n, _ = x.shape
    len_keep = int(n * (1 - mask_ratio))
    noise = torch.rand(b, n, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2])
    )
    mask = torch.ones([b, n], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore, ids_keep


def random_patch_masking(
    x: torch.Tensor, mask_ratio: float = 0.6, patch_size: int = 16
) -> torch.Tensor:
    """Apply random masking to image patches."""
    b, c, h, w = x.shape
    assert h % patch_size == 0 and w % patch_size == 0
    num_patches = (h // patch_size) * (w // patch_size)
    num_mask = int(mask_ratio * num_patches)
    x_masked = x.clone()
    for i in range(b):
        patch_indices = torch.randperm(num_patches)[:num_mask]
        mask = torch.ones(num_patches, device=x.device)
        mask[patch_indices] = 0
        mask = mask.view(h // patch_size, w // patch_size)
        mask = mask.repeat_interleave(patch_size, dim=0).repeat_interleave(
            patch_size, dim=1
        )
        x_masked[i] *= mask.unsqueeze(0)
    return x_masked


def setup_device_and_model(
    model: nn.Module, device: torch.device, logger: Any
) -> Tuple[nn.Module, torch.device]:
    """Setup device and wrap model with DataParallel if multiple GPUs available."""

    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s) available")

        if gpu_count > 1:
            logger.info(
                f"Using DataParallel with {gpu_count} GPUs: {[torch.cuda.get_device_name(i) for i in range(gpu_count)]}"
            )
            model = nn.DataParallel(model)
            # Set primary device
            device = torch.device(f"cuda:{model.device_ids[0]}")
        else:
            logger.info(f"Using single GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda:0")
    else:
        logger.info("CUDA not available, using CPU")
        device = torch.device("cpu")

    model = model.to(device)
    return model, device


def extract_visual_tensor(outputs: Any) -> torch.Tensor:
    """
    Simplified tensor extraction - assumes model outputs proper shapes
    """
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, dict):
        # Return first available tensor in priority order
        for key in ["recon", "output", "decoded", "logits"]:
            if key in outputs:
                return torch.Tensor(outputs[key])
    return torch.tensor(outputs)  # Final fallback


def _handle_tuple_list_batch(
    batch: Union[Tuple[Any, ...], List[Any]], device: str
) -> List[torch.Tensor]:
    """Handle tuple or list batch formats."""
    if len(batch) == 2 and all(isinstance(b, torch.Tensor) for b in batch):
        return [batch[0].to(device).float(), batch[1].to(device).float()]
    if len(batch) == 3 and all(isinstance(b, torch.Tensor) for b in batch[:2]):
        # Ignore label for contrastive training
        return [batch[0].to(device).float(), batch[1].to(device).float()]
    return []


def _handle_tensor_batch(batch: torch.Tensor, device: str) -> List[torch.Tensor]:
    """Handle tensor batch format."""
    if batch.ndim == 5 and batch.size(1) == 2:
        return [batch[:, 0].to(device).float(), batch[:, 1].to(device).float()]
    return []


def _handle_dict_batch(batch: Dict[str, Any], device: str) -> List[torch.Tensor]:
    """Handle dictionary batch format."""
    return [batch["view1"].to(device).float(), batch["view2"].to(device).float()]


def _handle_nested_batch(
    batch: Union[List[Any], Tuple[Any, ...]], device: str
) -> List[torch.Tensor]:
    """Handle nested batch format with multiple items."""
    view1_batch, view2_batch = [], []
    for item in batch:
        if isinstance(item, (tuple, list)):
            view1_batch.append(item[0])
            view2_batch.append(item[1])
        elif isinstance(item, dict):
            view1_batch.append(item["view1"])
            view2_batch.append(item["view2"])
    return [
        torch.stack(view1_batch).to(device).float(),
        torch.stack(view2_batch).to(device).float(),
    ]


def unpack_views_from_batch(
    batch: Union[torch.Tensor, Dict[str, torch.Tensor], List[Any], Tuple[Any, ...]],
    device: str,
) -> List[torch.Tensor]:
    """
    Unpack two augmented views from a batch for contrastive/self-supervised learning.
    Handles various batch formats (tuple, list, dict, tensor).
    """
    # Handle tuple or list format
    if isinstance(batch, (tuple, list)):
        result = _handle_tuple_list_batch(batch, device)
        if result:
            return result
        return _handle_nested_batch(batch, device)

    # Handle tensor format
    if isinstance(batch, torch.Tensor):
        result = _handle_tensor_batch(batch, device)
        if result:
            return result
        else:
            raise TypeError(f"Unsupported tensor batch shape: {batch.shape}")

    # Handle dictionary format
    if isinstance(batch, dict):
        return _handle_dict_batch(batch, device)

    raise TypeError(f"Unsupported batch type: {type(batch)}")
