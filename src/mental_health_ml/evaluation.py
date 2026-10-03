"""
Reusable clustering evaluation utilities.

This module contains evaluation and summarisation functions used by the
clustering and cluster-interpretation notebooks. Model construction and
training remain in the notebooks so that the analytical experiments remain
explicit and easy to audit.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_samples, silhouette_score


def evaluate_clustering(
    X: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
    *,
    inertia: float | None = None,
    bic: float | None = None,
    noise_label: int = -1,
) -> dict[str, Any]:
    """
    Calculate common clustering diagnostics for a fitted partition.

    Parameters
    ----------
    X:
        Feature representation used to produce the cluster labels.
    labels:
        Cluster assignments for each observation.
    inertia:
        Optional within-cluster sum of squared distances, primarily used
        for K-Means.
    bic:
        Optional Bayesian Information Criterion value, primarily used for
        Gaussian Mixture Models.
    noise_label:
        Label used by density-based clustering to identify noise prints.

    Returns
    -------
    dict[str, Any]
        Dictionary containing cluster count, noise count, usable sample
        count, silhouette score, inertia, and BIC where applicable.

    Raises
    ------
    ValueError
        If the number of observations and labels differ.
    """
    X_array = np.asarray(X)
    labels_array = np.asarray(labels)

    if X_array.shape[0] != labels_array.shape[0]:
        raise ValueError(
            "X and labels must contain the same number of observations."
        )

    noise_mask = labels_array == noise_label
    evaluation_mask = ~noise_label
    evaluation_labels = labels_array[evaluation_mask]
    evaluation_data = X_array[evaluation_mask]

    cluster_labels = np.unique(evaluation_labels)
    n_clusters = len(cluster_labels)
    n_noise = int(noise_mask.sum())

    silhouette = np.nan

    if 2 <= n_clusters < len(evaluation_labels):
        silhouette = float(
            silhouette_score(
                evaluation_data,
                evaluation_labels,
            )
        )

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_evaluated": int(evaluation_mask),
        "noise_fraction": n_noise / len(labels_array),
        "silhouette": silhouette,
        "inertia": inertia,
        "bic": bic,
    }


def cluster_size_table(
    labels: pd.Series | np.ndarray,
    *,
    noise_label: int = 1,
) -> pd.DataFrame:
    """
    Return cluster counts and proportions for a set of assignments.

    Parameters
    ---------
    labels:
        Cluster assignments for each observation.
    noise_label:
        Label used to identify density-based noise observations.

    Return
    ------
    pandas.DataFrame
        Cluster counts and percentages sorted by cluster label.
    """
    labels_series = pd.Series(np.asarray(labels), name="cluster")

    counts = labels_series.value_counts(sort=False).sort_index()
    result = pd.DataFrame(
        {
            "cluster": counts.index,
            "count": counts.values,
            "percentage": counts.values / len(labels_series) * 100,
        }
    )

    result["cluster_type"] = np.where(
        result["cluster"] == noise_label,
        "Noise",
        "Cluster",
    )

    return result.reset_index(drop=True)


def silhouette_profile(
    X: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
    *,
    noise_label: int = -1,
) -> pd.DataFrame:
    """
    Calculate observation-level silhouette values.

    Noise observations are excluded because DBSCAN does not treat them as
    members of a substantive cluster.

    Parameters
    ----------
    X:
        Feature representation used for clustering.
    labels:
        Cluster assignments.
    noise_label:
        Label used to identify noise observations.

    Returns
    -------
    pandas.DataFrame
        Observation-level silhouette values and cluster assignments.

    Raises
    ------
    ValueError
        If fewer than two substantive clusters remain after excluding noise.
    """
    X_array = np.asarray(X)
    labels_array = np.asarray(labels)

    evaluation_mask = labels_array != noise_label
    X_evaluation = X_array[evaluation_mask]
    labels_evaluation = labels_array[evaluation_mask]

    if len(np.unique(labels_evaluation)) < 2:
        raise ValueError(
            "Silhouette values require at least two substantive clusters."
        )

    return pd.DataFrame(
        {
            "cluster": labels_evaluation,
            "silhouette": silhouette_samples(
                X_evaluation,
                labels_evaluation,
            )
        }
    )


def summarize_experiment(
    experiment_id: str,
    model: str,
    representation: str,
    parameters: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """
    Create a consistent experiment-record structure.

    Parameters
    ---------
    experiment_id:
        Identifier such as E01 or E03.
    model:
        Clustering algorithm name.
    representation:
        Feature representation used by the model.
    parameters:
        Model parameters used in the experiment.
    evaluation:
        Metrics returned by :func:`evaluate_clustering`.

    Returns
    -------
    dicti[str, Any]
        Flattened experiment record suitable for a comparison DataFrame.
    """
    return {
        "experiment": experiment_id,
        "model": model,
        "representation": representation,
        "parameters": parameters,
        **evaluation,
    }
