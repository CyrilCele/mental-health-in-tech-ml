"""Utilities for loading and validating project datasets."""

import zipfile
from pathlib import Path

import pandas as pd


def load_dataset(path: Path, target_name: str | None = None) -> pd.DataFrame:
    """Load a dataset from a CSV file or a ZIP archive containing a CSV file.

    Args:
        path (Path): The path to the CSV file or ZIP archive.
        target_name (str, optional): The name of the specific CSV file inside the ZIP
                                    archive. If None and it's a ZIP, it assumes the
                                    archive contains only one CSV file.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")

    try:
        # Check if the file is a ZIP archive
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as z:
                # If target_name isn't provided, try to find the first CSV file automatically
                if target_name is None:
                    csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                    if not csv_files:
                        raise ValueError(f"No CSV files found inside the ZIP archive {path}.")
                    target_name = csv_files[0]  # Take the first CSV file found

                # Verify the requested file actually exists inside the ZIP
                if target_name not in z.namelist():
                    raise KeyError(
                        f"'{target_name}' not found in the ZIP archive."
                        f"Available files: {z.namelist()}"
                    )

                # Open and read the specific CSV file from the ZIP
                with z.open(target_name) as f:
                    df = pd.read_csv(f)

        else:
            # Handle standard, uncompromised CSV files
            df = pd.read_csv(path)

    except Exception as e:
        raise ValueError(f"Error loading the dataset from {path}: {e}")

    return df
