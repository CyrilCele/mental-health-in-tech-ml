"""Reusable data-cleaning functions for the mental-health survey dataset."""

import pandas as pd


def clean_age(
    data: pd.DataFrame,
    column: str = "What is your age?",
    minimum_age: int = 18,
    maximum_age: int = 75,
) -> pd.DataFrame:
    """
    Replace ages outised the accepted working-age range with missing values.

    Parameters
    ----------
    data:
        Survey data containing the age variable.
    column:
        Name of the age column.
    minimum_age:
        Inclusive lower bound for valid ages.
    maximum_age:
        Inclusive upper bound for valid ages.

    Returns
    -------
    pandas.DataFrame
        Copy of the input data with invalid ages represented as missing values.

    Raises
    ------
    KeyError
        If the specified age column is not present.
    ValueError
        If the minimum age is greater than the maximun age.
    """
    if column not in data.columns:
        raise KeyError(f"Age column not found: {column}")

    if minimum_age > maximum_age:
        raise ValueError("Minimum age must not exceed maximim age.")

    cleaned = data.copy()

    cleaned[column] = cleaned[column].where(
        cleaned[column].between(minimum_age, maximum_age)
    )

    return cleaned
