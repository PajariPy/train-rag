"""Download the HERB dataset from Hugging Face."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

HERB_REPO_ID = "Salesforce/HERB"
DEFAULT_DATA_DIR = Path("data")


def download_herb(data_dir: Path) -> Path:
    """Download the HERB metadata and product files.

    Args:
        data_dir: Directory to place the dataset in.

    Returns:
        The directory the dataset was downloaded to.
    """
    local_dir = snapshot_download(
        repo_id=HERB_REPO_ID,
        repo_type="dataset",
        local_dir=data_dir,
        allow_patterns=["README.md", "metadata/*", "products/*"],
    )

    return Path(local_dir)


def main() -> None:
    """Run the download from the command line."""
    parser = argparse.ArgumentParser(description="Download the HERB dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    path = download_herb(args.data_dir)
    print(f"HERB downloaded to {path}")
