"""
Reusable dimensionality-reduction functions.

The functions in this module provide reproducible PCA and MDS operstaions for
the Mental Health in Technology clustering project.

Notebook 03 remains responsible for the analytical decisions and interpretation
of dimensionality reduction. This module only contains reusable computational
operations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import MDS


def fit_pca(
    data: pd.DataFrame,
    n_components: int | None = None,
) -> tuple[PCA, pd.DataFrame]:
    """
    Fit PCA and return the fitted transformer and transformed data.

    Parameters
    ----------
    data:
        Numerical feature matrix used as the PCA input.
    n_components:
        Number of components to retain. If None, all components are
        fitted.

    Returns
    -------
    tuple[PCA, pd.DataFrame]
        Fitted PCA transformer and transformed feature matrix.
    """
    model = PCA(n_components=n_components)
    transformed = model.fit_transform(data)

    columns = [
        f"PC{i}"
        for i in range(1, model.n_components_ + 1)
    ]

    return model, pd.DataFrame(
        transformed,
        index=data.index,
        columns=columns
    )


def build_pca_variance_table(
    model: PCA,
) -> pd.DataFrame:
    """
    Create a table containing individual and cumulative PCA variance.

    Parameters
    ----------
    model:
        Fitted PCA transformer.

    Returns
    -------
    pd.DataFrame
        Component-level explained-variance information.
    """
    return pd.DataFrame(
        {
            "component": np.arange(
                1,
                len(model.explained_variance_ratio_) + 1,
            ),
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_explained_variance": (
                model.explained_variance_ratio_.cumsum()
            ),
        }
    )


def fit_mds(
    data: pd.DataFrame,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[MDS, pd.DataFrame]:
    """
    Fit metric MDS and return the transformer and two-dimensional data.

    Parameters
    ----------
    data:
        Numerical feature matrix used as the MDS input.
    n_components:
        Number of dimensions in the resulting representation.
    random_state:
        Random seed used to make the MDS solution reproducible.

    Returns
    -------
    tuple[MDS, pd.DataFrame]
        Fitted MDS estimator and transformed feature matrix.
    """
    model = MDS(
        n_components=n_components,
        random_state=random_state,
        normalized_stress="auto",
    )

    transformed = model.fit_transform(data)

    columns = [
        f"MDS{i}"
        for i in range(1, n_components + 1)
    ]

    return model, pd.DataFrame(
        transformed,
        index=data.index,
        columns=columns
    )
