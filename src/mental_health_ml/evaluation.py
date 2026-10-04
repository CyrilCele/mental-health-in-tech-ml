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

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def evaluate_clustering(
    X: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
    *,
    inertia: float | None = None,
    bic: float | None = None,
    aic: float | None = None,
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
    aic:
        Optional Akaike Information Criterion value, primarily used for
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
        "aic": aic,
    }


def calinski_harabasz_index(
    X: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
    *,
    noise_label: int = -1,
) -> float:
    """
    Calculate the Calinski-Harabasz index for substantive clusters.

    DBSCAN noise observations are excluded so that the metric describes the
    same substantive cluster partition used by the project's silhouette
    calculation.

    Parameters
    ----------
    X:
        Feature representation used for clustering.
    labels:
        Cluster assignments.
    noise_label:
        Label identifying observations excluded as noise.

    Returns
    -------
    float
        Calinski-Harabasz index. Higher values indicate stronger separation
        relative to within-cluster dispersion.

    Raises
    ------
    ValueError
        If fewer than two substantive clusters are available.
    """
    X_array = np.asarray(X)
    labels_array = np.asarray(labels)
    evaluation_mask = labels_array != noise_label

    X_evaluation = X_array[evaluation_mask]
    labels_evaluation = labels_array[evaluation_mask]

    if len(np.unique(labels_evaluation)) < 2:
        raise ValueError(
            "The Calinski-Harabasz index requires at least two substantive clusters."
        )

    return float(
        calinski_harabasz_score(
            X_evaluation,
            labels_evaluation,
        )
    )


def cophenetic_correlation(
    linkage_matrix: np.ndarray,
    X: pd.DataFrame | np.ndarray,
) -> float:
    """
    Calculate the cophenetic correlation of a hierarchical solution.

    Parameters
    ----------
    linkage_matrix:
        Linkage matrix generated by a SciPy hierarchical clustering method.
    X:
        Original feature representation from which pairwise distances are
        calculated.

    Returns
    -------
    float
        Correlation between original pairwise distances and cophenetic
        distances represented by the dendrogram.
    """
    correlation, _ = cophenet(
        linkage_matrix,
        pdist(np.asarray(X)),
    )
    return float(correlation)


def k_distance_profile(
    X: pd.DataFrame | np.ndarray,
    *,
    n_neighbors: int,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Calculate ordered kth-nearest-neighbour distances for DBSCAN analysis.

    Parameters
    ----------
    X:
        Feature representation used for DBSCAN.
    n_neighbors:
        Neighbour rank used for the distance profile. The resulting distance
        is the distance to the kth neighbour.
    metric:
        Distance metric passed to :class:`sklearn.neighbors.NearestNeighors`.

    Returns
    -------
    pandas.Dataframe
        Observation order and corresponding kth-nearest-neighbor distance.

    Raises
    ------
    ValueError
        If ``n_neighbors`` is outside the valid range for the data.
    """
    X_array = np.asarray(X)

    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")

    if not 1 <= n_neighbors < len(X_array):
        raise ValueError(
            "n_neighbors must be between 1 and one less than the number "
            "of observations."
        )

    # NearestNeighbors includes each observation itself at distance zero.
    # Request one additional neighbour so the requested rank refers to an
    # actual neighbouring observation rather than the observation itself.
    model = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric=metric,
    )

    model.fit(X_array)

    distances, _ = model.kneighbors(X_array)
    ordered_distances = np.sort(distances[:, 1])

    return pd.DataFrame(
        {
            "observation_order": np.arange(1, len(ordered_distances) + 1),
            "k_distance": ordered_distances,
        }
    )


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


def estimate_knee_from_k_distance(
    k_distance_profile: pd.DataFrame,
) -> float:
    """
    Estimate an ``eps`` knee from an ordered k-distance curve.

    The method normailizes observation order and k-distance values to the unit
    square and identifies the point with the largest perpendicular distance
    from the straight line joining the first and last points. This provides a
    deterministic geometric knee estimate for DBSCAN parameter exp;oration.

    Parameters
    ----------
    k_distance_profile:
        DataFrame returned by :func:`k_distance_profile`.

    Returns
    -------
    float
        Estimated K-distance at the detected knee.

    Raises
    ------
    ValueError
        If the profile contains fewer than three observations or non-finite
        distances.
    """
    required_columns = {"observation_order", "k_distance"}

    if not required_columns.issubset(k_distance_profile.columns):
        raise ValueError(
            "k_distance_profile must contain observation_order and k_distance."
        )

    if len(k_distance_profile) < 3:
        raise ValueError(
            "At least three observations are required to estimate a knee."
        )

    distances = k_distance_profile["k_distance"].to_numpy(dtype=float)

    if not np.isfinite(distances).all():
        raise ValueError(
            "K-distance values must be finite."
        )

    x = np.linspace(0.0, 1.0, len(distances))
    y_min = distances.min()
    y_max = distances.max()

    if np.isclose(y_min, y_max):
        return float(y_min)

    y = (distances - y_min) / (y_max - y_min)

    # Calculate perpendicular distance from every point to the line joining
    # the first and last normalized observations.
    start = np.array([x[0], y[0]])
    end = np.array(x[-1], y[-1])
    line_vector = end - start

    distances_to_line = np.abs(
        line_vector[0] * (start[1] - y)
        - (start[0] - x) * line_vector[1]
    ) / np.linalg.norm(line_vector)

    knee_index = int(np.argmax(distances_to_line))

    return float(distances[knee_index])
