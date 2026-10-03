"""
Reusable clustering evaluation utilities.

This module contains project-owned functions for evaluating and summarising
clustering experiments. The clustering algorithms themselves remain in the
modeling notebook so that construction and analytical decisions remain
explicit and auditable.
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
    Calculate standard diagnostics for a clustering partition.

    Parameters
    ----------
    X:
        Feature representation used to generate the cluster assignments.
    labels:
        Cluster assignment for each observation.
    inertia:
        Optional within-cluster sum of squared distances, primarily used
        for K-Means.
    bic:
        Optional Bayesian Information Criterion value, primarily used for
        Gaussian Mixture Models.
    noise_label:
        Label used to identify noise observations in density-based
        clustering. DBSCAN uses ``-1`` by convention.

    Returns
    ------
    dict[str, Any]
        Cluster count, noise count, evaluated sample count, noise fraction,
        silhouette score, inertia, and BIC where applicable.

    Raises
    ------
    ValueError
        If the number of observations and labels do not match.
    """
    X_array = np.asarray(X)
    labels_array = np.asarray(labels)

    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")

    if labels_array.ndim != 1:
        raise ValueError("Labels must be a one-dimensional array.")

    if X_array.shape[0] != labels_array.shape[0]:
        raise ValueError(
            "X and labels must contain the same number of observations."
        )

    # Exclude DBSCAN noise observations from silhouette evaluation.
    noise_mask = labels_array == noise_label
    evaluation_mask = labels_array != noise_label

    evaluation_data = X_array[evaluation_mask]
    evaluation_labels = labels_array[evaluation_mask]

    cluster_labels = np.unique(evaluation_labels)
    n_clusters = len(cluster_labels)
    n_evaluated = int(evaluation_mask.sum())
    n_noise = int(noise_mask.sum())

    silhouette = np.nan

    # A silhouette score requires at least two substantive clustes and
    # at least one observation in each evaluated cluster.
    if 2 <= n_clusters < n_evaluated:
        silhouette = float(
            silhouette_score(
                evaluation_data,
                evaluation_labels,
            )
        )

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_evaluated": n_evaluated,
        "noise_fraction": n_noise / len(labels_array),
        "silhouette": silhouette,
        "inertia": inertia,
        "bic": bic,
    }


def cluster_size_table(
    labels: pd.Series | np.ndarray,
    *,
    noise_label: int = -1,
) -> pd.DataFrame:
    """
    Return cluster counts and proportions.

    Parameters
    ---------
    labels:
        Cluster assignments for each observation.
    noise_label:
        Label used to identify noise observations.

    Returns
    -------
    pandas.DataFrame
        Cluster counts, percentages, and cluster type.
    """
    labels_series = pd.Series(
        np.asarray(labels),
        name="cluster",
    )

    counts = (
        labels_series
        .value_counts(sort=False)
        .sort_index()
    )

    result = pd.DataFrame(
        {
            "cluster": counts.index,
            "count": counts.values,
            "percentage": (
                counts.values / len(labels_series) * 100  # type: ignore
            ),
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
    Calculate observation-levl silhouette coefficients.

    Parameters
    ----------
    X:
        Feature representation used for clustering.
    labels:
        Cluster assignments.
    noise_label:
        Label used to identify density-based noise observations.

    Returns
    -------
    pandas.DataFrame
        Observation-level silhouette coefficients and cluster labels.

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
            ),
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
    Create a consistent record for a clustering experiment.

    Parameters
    ----------
    experiment_id:
        Experiment identifier such as ``E01``.
    model:
        Clustering algorithm name.
    representation:
        Feature representation used by the model.
    parameters:
        Model parameters used in the experiment.
    evaluation:
        Evaluation dictionary returned by
        :func:`evaluate_clustering`.

    Returns
    -------
    dict[str, Any]
        Flattened experiment record suitable for a comparison DataFrame.
    """
    return {
        "experiment": experiment_id,
        "model": model,
        "representation": representation,
        "parameters": parameters,
        **evaluation,
    }
