"""
Reusable unspuervised feature-selection functions.

The functions in this module support feature-selection decisions used during
the dimensionality-reduction stage of the Mental Health in Technology project.

The selection precedures are deliberately unsupervised because the project
does not define a supervised production target for the clustering analysis.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def remove_zero_variance_features(
        data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove features with no variation across respondents.

    A feature with zero variance contains the same value for every respondent
    and therefore cannot contribute to separating observations in an
    unsupervised analysis.

    Parameters
    ----------
    data:
        Numerical analytical feature matrix.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]:
        The feature matrix after zero-variance removal and the names of the
        removed features.

    Raises
    ------
    ValueError
        If the input contains no columns.
    """
    if data.empty:
        raise ValueError(
            "The input feature matrix must contain at least one column."
        )

    zero_variance_columns = data.columns[
        data.nunique(dropna=False) <= 1
    ].tolist()

    return (
        data.drop(columns=zero_variance_columns),
        zero_variance_columns,
    )


def find_highly_correlated_features(
        data: pd.DataFrame,
        threshold: float = 0.95,
) -> list[tuple[str, str, float]]:
    """
    Identify redundant feature pairs using absolute Pearson correlation.

    Only the upper triangle of the correlation matrix is inspected so that
    each feature pair is reported once.

    Parameters
    ----------
    data:
        Numerical analytical feature matrix.
    threshold:
        Absolute correlation above which a feature pair is considered highly
        redundant.

    Returns
    -------
    list[tuple[str, str, float]]:
        Feature pairs exceeding the specified absolute-correlation threshold.

    Raises
    ------
    ValueError
        If the correlation thrshold is outside the interval (0, 1)
    """
    if not 0 < threshold < 1:
        raise ValueError(
            "Threshold must be greater than 0 and less than 1."
        )

    correlation_matrix = data.corr().abs()

    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1,
        ).astype(bool)
    )

    correlated_pairs = [
        (column, row, upper_triangle.loc[row, column])
        for column in upper_triangle.columns
        for row in upper_triangle.index
        if pd.notna(upper_triangle.loc[row, column])
        and upper_triangle.loc[row, column] > threshold
    ]

    return sorted(
        correlated_pairs,
        key=lambda pair: pair[2],
        reverse=True,
    )


def remove_redundant_features(
        data: pd.DataFrame,
        threshold: float = 0.95,
) -> tuple[pd.DataFrame, list[str], list[tuple[str, str, float]]]:
    """
    Remove redundant features from highly correlated feature pairs.

    When two features exceed the specified absolute-correlation threshold,
    the later feature in the deterministic column order is removed. This
    produces a reproducible feature-selection result without using a target
    variable.

    Parameters
    ----------
    data:
        Numerical analytical feature matrix.
    threshold:
        Absolute correlation above which features are considered redundant.

    Returns
    -------
    tuple[pd.DataFrame, list[str], list[tuple[str, str, float]]]:
        The reduced feature matrix, removed feature names, and correlated
        feature pairs used to make the decision.
    """
    correlated_pairs = find_highly_correlated_features(
        data,
        threshold=threshold,
    )

    features_to_remove = {
        second_feature
        for first_feature, second_feature, _ in correlated_pairs
    }

    return (
        data.drop(columns=sorted(features_to_remove)),
        sorted(features_to_remove),
        correlated_pairs,
    )
