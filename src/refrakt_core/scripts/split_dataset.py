import os
import random
import shutil
from pathlib import Path


def rename_lr_images(lr_dir="../data/DIV2K/LR"):
    lr_path = Path(lr_dir)
    for file in lr_path.glob("*x2.png"):
        # Example: 0001x2.png → 0001.png
        new_name = file.name.replace("x2", "")
        new_path = file.parent / new_name
        file.rename(new_path)
        print(f"Renamed {file.name} → {new_name}")


def split_dataset(hr_dir, lr_dir, output_dir, split_ratio=0.8, seed=42):
    random.seed(seed)

    all_files = sorted(os.listdir(hr_dir))
    random.shuffle(all_files)

    split_idx = int(len(all_files) * split_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    for phase, file_list in zip(
        ["train", "val"], [train_files, val_files], strict=False
    ):
        hr_out = Path(output_dir) / phase / "HR"
        lr_out = Path(output_dir) / phase / "LR"
        hr_out.mkdir(parents=True, exist_ok=True)
        lr_out.mkdir(parents=True, exist_ok=True)

        for fname in file_list:
            shutil.copy(Path(hr_dir) / fname, hr_out / fname)
            shutil.copy(Path(lr_dir) / fname, lr_out / fname)


if __name__ == "__main__":
    # rename_lr_images()
    split_dataset(
        hr_dir="../data/DIV2K/HR",
        lr_dir="../data/DIV2K/LR",
        output_dir="../data/DIV2K/",
    )
