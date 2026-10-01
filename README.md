# Mental Health in Tech ML

An interpretable unsupervised machine-learning analysis of the OSMI Mental Health in Tech 2016 survey.

The project applies the CRISP-DM framework to investigate whether meaningful respondent patterns can be identified from survey responses concerning mental health and technology-related workplaces.

## Project Objective

The objective is to identify statistically meaningful groups of survey respondents based on patterns in their responses.

These groups are intended for analytical interpretation and organizational insight. They are not medical diagnoses and should not be used to make individual employment or clinical decisions.

## Analytical Approach

The project follows:

- CRISP-DM
- data-quality assessment
- feature engineering
- feature selection
- dimensionality reduction
- unsupervised clustering
- cluster evaluation
- cluster profiling
- reproducible documentation

### Clustering Methods

The project evaluates four clustering approaches:

- K-Means
- Gaussian Mixture Models
- Agglomerative Hierarchical Clustering
- DBSCAN

### Dimensionality Reduction

The project evaluates:

- Principal Component Analysis (PCA)
- Multidimensional Scaling (MDS)
- Locally Linear Embedding (LLE), where justified

The final analytical choices will be based on the observed data and experimental evidence.

## Dataset

The analysis uses the OSMI Mental Health in Tech 2016 survey dataset.

Source:

https://www.kaggle.com/osmi/mental-health-in-tech-2016

The raw dataset is not committed to this repository.

## Repository Structure

```text
mental-health-in-tech-ml/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_business_and_data_understanding.ipynb
│   ├── 02_data_preparation_and_feature_engineering.ipynb
│   ├── 03_dimensionality_reduction.ipynb
│   ├── 04_clustering_models.ipynb
│   └── 05_cluster_evaluation_and_interpretation.ipynb
├── src/
│   └── mental_health_ml/
│       ├── __init__.py
│       ├── data.py
│       ├── cleaning.py
│       ├── preprocessing.py
│       ├── feature_engineering.py
│       ├── feature_selection.py
│       ├── dimensionality_reduction.py
│       ├── download.py
│       ├── evaluation.py
│       └── visualization.py
├── tests/
│   ├── test_cleaning.py
│   ├── test_preprocessing.py
│   └── test_feature_engineering.py
├── reports/
│   └── figures/
└── docs/
    ├── methodology.md
    ├── data_dictionary.md
    └── limitations.md
```

## Environment

This project uses:

- Python
- uv
- scikit-learn
- pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Jupyter
- Kaggle API

`uv.lock` provides reproducible dependency resolution.

## Local Setup

After cloning the repository:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

Run code-quality checks:

```bash
uv run ruff check .
```

Format the Python source:

```bash
uv run ruff format .
```

Start Jupyter:

```bash
uv run jupyter lab
```

## Development Philosophy

The notebooks contain the analytical narrative and model experiments.

Reusable data-processing, feature-engineering, evaluation, and visualization logic belongs in `src/mental_health_ml/`.

Models are intentionally kept within the notebooks rather than being packaged as deployment services.

The project does not include:

- model-serving APIs
- inference services
- model registries
- Kubernetes
- MLOps infrastructure
- unnecessary cloud infrastructure

## Reproducibility

The analysis will use:

- explicit dependency versions
- reproducible preprocessing
- controlled random seeds where appropriate
- documented parameters
- documented dataset acquisition
- reusable Python modules
- automated tests for project-owned logic

## Ethical Considerations

Mental-health information is sensitive.

The analysis therefore treats clustering as a statistical segmentation technique rather than a diagnostic system.

Interpretations will explicitly consider:

- privacy
- stigma
- survey limitations
- representativeness
- demographic bias
- self-reporting
- correlation versus causation
- uncertainty in cluster assignment
- potential misuse of segmentation

## Status

**Project stage:** Repository initialization

The analytical results have not yet been produced.

The project will progess incrementally through the CRISP-DM lifecycle.
