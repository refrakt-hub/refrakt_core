import zipfile
from pathlib import Path


def unzip_div2k() -> None:
    zip_dir = Path("../zips")
    extract_dir = Path("../data/DIV2K")
    extract_dir.mkdir(parents=True, exist_ok=True)

    hr_zip = zip_dir / "DIV2K_train_HR.zip"
    lr_zip = zip_dir / "DIV2K_train_LR_unknown_X2.zip"

    with zipfile.ZipFile(hr_zip, "r") as zip_ref:
        zip_ref.extractall(extract_dir / "HR")

    with zipfile.ZipFile(lr_zip, "r") as zip_ref:
        zip_ref.extractall(extract_dir / "LR")


if __name__ == "__main__":
    unzip_div2k()
