"""
Reusable visualization functions for clustering analysis.

The functions in this module create analytical figures used by the
clustering and interpretation notebooks. The notebooks remain responsible
for deciding which figures are needed and how their results should be
interpreted.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram


def plot_metric_curve(
    results: pd.DataFrame,
    *,
    x: str,
    y: str,
    ax: plt.Axes,  # type: ignore
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """
    Plot a model-selection metric across a parameter range.

    Parameters
    ----------
    results:
        DataFrame containing the parameter and metric columns.
    x:
        Name of the parameter column.
    y:
        Name if the metric column.
    ax:
        Matplotlib axis on which to draw the figure.
    title:
        Figure title.
    xlabel:
        Horizontal-axis label.
    ylabel:
        Vertical-axis label.
    """
    sns.lineplot(
        data=results,
        x=x,
        y=y,
        marker="o",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def plot_silhouette_profile(
    silhouette_data: pd.DataFrame,
    *,
    ax: plt.Axes,  # type: ignore
    title: str,
) -> None:
    """
    Plot observation-level silhouette coefficients by cluster.

    Parameters
    ----------
    silhouette_data:
        DataFrame containing ``cluster`` and ``silhouette`` columns.
    ax:
        Matplotlib axis on which to draw the figure.
    title:
        Figure title.
    """
    y_lower = 10

    for cluster in sorted(
        silhouette_data["cluster"].unique()
    ):
        values = np.sort(
            silhouette_data.loc[
                silhouette_data["cluster"] == cluster,
                "silhouette",
            ].to_numpy()
        )

        size = len(values)
        y_upper = y_lower + size

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            values,
            alpha=0.7,
        )

        ax.text(
            -0.05,
            y_lower + 0.5 * size,
            str(cluster),
            fontsize=9,
        )

        y_lower = y_upper + 10

    ax.axvline(
        silhouette_data["silhouette"].mean(),
        linestyle="--",
        linewidth=1,
        label="Mean silhouette",
    )

    ax.set_title(title)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.legend()
    ax.grid(alpha=0.2)


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    *,
    ax: plt.Axes,  # type: ignore
    title: str,
    truncate_level: int | None = None,
) -> None:
    """
    Plot a hierarchical-clustering dendrogram.

    Parameters
    ----------
    linkage_matrix:
        Linkage matrix produced by SciPy hierarchical clustering.
    ax:
        Matplotlib axis on which to draw the dendrogram.
    title:
        Figure title.
    truncate_level:
        Optional number of hierarchy levels to display.
    """
    dendrogram_kwargs = {
        "ax": ax,
        "no_labels": True,
    }

    if truncate_level is not None:
        dendrogram_kwargs.update(
            {
                "truncate_mode": "level",
                "p": truncate_level,
            }
        )

    dendrogram(
        linkage_matrix,
        **dendrogram_kwargs,
    )

    ax.set_title(title)
    ax.set_xlabel("Observations / merged groups")
    ax.set_ylabel("Cluster distance")


def plot_cluster_projection(
    X_2d: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
    *,
    ax: plt.Axes,  # type: ignore
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """
    Plot cluster assignments in a two-dimensional representation.

    Parameters
    ----------
    X_2d:
        Two-dimensional representation used for visualization.
    labels:
        Cluster assignments.
    ax:
        Matplotlib axis on which to draw the figure.
    title:
        Figure title.
    x_label:
        Horizontal-axis label.
    y_label:
        Vertical-axis label.
    """
    data = pd.DataFrame(
        {
            "x": np.asarray(X_2d)[:, 0],
            "y": np.asarray(X_2d)[:, 1],
            "cluster": np.asarray(labels),
        }
    )

    sns.scatterplot(
        data=data,
        x="x",
        y="y",
        hue="cluster",
        style="cluster",
        alpha=0.75,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.2)


def plot_cluster_sizes(
    cluster_sizes: pd.DataFrame,
    *,
    ax: plt.Axes,  # type: ignore
    title: str,
) -> None:
    """
    Plot the number of observations assigned to each cluster.

    Parameters
    ----------
    cluster_sizes:
        DataFrame returned by :func:`cluster_size_table`.
    ax:
        Matplotlib axis on which to draw the figure.
    title:
        Figure title.
    """
    sns.barplot(
        data=cluster_sizes,
        x="cluster",
        y="count",
        hue="cluster_type",
        dodge=False,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of observations")
    ax.grid(axis="y", alpha=0.2)
