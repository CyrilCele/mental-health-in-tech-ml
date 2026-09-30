"""
Download the raw OSMI Mental Health in Tech survey dataset.

This module is responsibles only for acquiring the original survey data
from Kaggle. It does not clean, transform, or analyse the dataset.

The downloaded CSV is intentionally kept unchanged in ``data/raw/`` so
that the analytical notebooks can perform and document the complete
data-understanding and data-preparation process.
"""

from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle dataset identifier for the OSMI Mental Health in Tech 2016 survey.
DATASET_NAME = "osmi/mental-health-in-tech-2016"

# Original survey CSV filename supplied by the Kaggle dataset.
DATASET_FILE = "mental-heath-in-tech-2016_20161114.csv"

# Project location for immutable raw input data.
RAW_DATA_DIR = Path("data/raw")


def download_dataset(
    dataset_name: str = DATASET_NAME,
    dataset_file: str = DATASET_FILE,
    download_path: Path = RAW_DATA_DIR,
) -> None:
    """
    Download the original survey CSV from Kaggle.

    Args:
        dataset_name (str): Kaggle dataset identifier in ``owner/dataset`` format.
        dataset_file (str): Exact filename of the survey CSV within the Kaggle dataset.
        download_path (Path): Local directory where the raw CSV will be stored.

    Raises:
        RuntimeError: If Kaggle authentication fails.
    """
    print("\nAuthenticating with Kaggle...")

    api = KaggleApi()

    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle authentication failed. Configure Kaggle API credentials "
            "before downloading the dataset."
        ) from exc

    # Create the raw-data directory without modifying existing raw files.
    download_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading '{dataset_file}' to '{download_path}'...")

    api.dataset_download_file(
        dataset_name,
        dataset_file,
        path=str(download_path),
        force=False,  # Do not overwrite existing files.
        quiet=False  # Show download progress.
    )

    print("Dataset download complete. The raw CSV is available in 'data/raw/'.\n")


if __name__ == "__main__":
    download_dataset()
