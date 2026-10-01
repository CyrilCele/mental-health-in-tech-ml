"""Reusable feature-engineering functions for the survey dataset."""

import pandas as pd


def engineer_work_position_features(
    data: pd.DataFrame,
    column: str = "Which of the following best describes your work position?",
    prefix: str = "work_position__",
) -> pd.DataFrame:
    """
    Decompose the multi-response work-position field into binary indicators.

    Each role selected by a respondent becomes an individual binary feature.
    The original multi-repsonse column is removed from the returned dataframe.

    Parameters
    ----------
    data:
        Survey data containing the multi-response work-position variable.
    column:
        Name of the multi-response work-position column.
    prefix:
        Prefix applied to generated role-indicator columns.

    Returns
    -------
    KeyError
        If the specified work-position column is not present.
    """
    if column not in data.columns:
        raise KeyError(f"Work-position column not found: {column}")

    engineered = data.copy()

    role_features = (
        engineered[column]
        .fillna("")
        .astype(str)
        .str.get_dummies(sep="|")
        .rename(columns=lambda value: f"{prefix}{value.strip()}")
    )

    return (
        engineered.drop(columns=column)
        .join(role_features)
    )
