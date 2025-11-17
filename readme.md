# Sparse Partial Least Squares (Sparse PLS) Regression

[![PyPI version](https://badge.fury.io/py/sparse-pls.svg)](https://pypi.org/project/sparse-pls/)
[![Python Versions](https://img.shields.io/pypi/pyversions/sparse-pls.svg)](https://pypi.org/project/sparse-pls/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/yajeddig/SPARSE_PLS/actions/workflows/test.yml/badge.svg)](https://github.com/yajeddig/SPARSE_PLS/actions/workflows/test.yml)
[![PyPI Downloads](https://img.shields.io/pypi/dm/sparse-pls.svg)](https://pypi.org/project/sparse-pls/)

---

## Introduction

The **Sparse Partial Least Squares (Sparse PLS)** project implements a regression model that combines dimensionality reduction with variable selection. This model is particularly useful for high-dimensional datasets where interpretability and feature selection are important. By incorporating sparsity into the Partial Least Squares (PLS) regression framework, the model selects the most relevant features while capturing the underlying relationship between predictors and responses.

---

## Features

- **Dimensionality Reduction**: Reduces the dataset to a lower-dimensional latent space.
- **Variable Selection**: Introduces sparsity to select the most relevant features.
- **Customizable Scaling**: Supports data preprocessing with different scaling methods.
- **Hyperparameter Optimization**: Includes methods for cross-validation and hyperparameter tuning.
- **Scikit-learn Compatibility**: Designed to integrate seamlessly with scikit-learn's ecosystem.

---

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Example Workflow](#example-workflow)
- [Classes and Methods](#classes-and-methods)
  - [SparsePLS](#sparsepls-class)
  - [DataPreprocessor](#datapreprocessor-class)
- [Mathematical Background](#mathematical-background)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

### From PyPI (Recommended)

```bash
pip install sparse-pls
```

### From Source (Development)

```bash
git clone https://github.com/yajeddig/SPARSE_PLS.git
cd SPARSE_PLS
pip install -r requirements.txt
pip install -e .
```

---

## Dependencies

- **Python 3.12.4**
- **numpy==1.26.4**
- **pandas==2.2.2**
- **scikit-learn==1.5.1**
- **scipy==1.13.1**
- **joblib==1.4.2**

Ensure that all dependencies are installed before running the project. The `requirements.txt` file contains all the necessary packages and their versions.

---

## Usage

### Example Workflow

```python
import pandas as pd
from model import SparsePLS
from preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv('your_dataset.csv')
X = data.drop('target_column', axis=1)
y = data['target_column']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the model
model = SparsePLS(n_components=2, alpha=0.5)

# Fit the model
model.fit(X_train, y_train)

# Transform the data
X_scores = model.transform(X_test)

# Predict target values
y_pred = model.predict(X_test)

# Optimize hyperparameters
param_grid = {
    'n_components': [1, 2, 3],
    'alpha': [0.1, 0.5, 1.0]
}
model.optimize_parameters(
    X_train, y_train,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Access selected variables
selected_features = model.feature_names_in_[model.selected_variables_]
print("Selected Features:")
print(selected_features)
```

---

## Classes and Methods

### SparsePLS Class

#### Overview

The `SparsePLS` class implements the Sparse Partial Least Squares regression model. It performs both dimensionality reduction and variable selection by introducing sparsity into the PLS framework.

#### Initialization Parameters

- `n_components` (int, default=2): Number of components to extract.
- `alpha` (float, default=1.0): Regularization parameter controlling sparsity.
- `max_iter` (int, default=500): Maximum number of iterations.
- `tol` (float, default=1e-6): Tolerance for convergence.
- `scale` (bool, default=True): Whether to scale the data.
- `scale_method` (str, default='standard'): Scaling method to use.
- `**kwargs`: Additional keyword arguments for the scaler.

#### Key Methods

- `fit(X, Y)`: Fits the model to the data.
- `transform(X)`: Transforms new data using the learned components.
- `fit_transform(X, Y)`: Fits the model and transforms the data.
- `predict(X)`: Predicts target values for new data.
- `optimize_parameters(X, Y, param_grid, cv, scoring, n_jobs, verbose, return_models)`: Optimizes hyperparameters using cross-validation.

#### Attributes

- `x_weights_`: Weights for predictors.
- `y_weights_`: Weights for responses.
- `x_loadings_`: Loadings for predictors.
- `y_loadings_`: Loadings for responses.
- `x_scores_`: Scores for predictors.
- `y_scores_`: Scores for responses.
- `coef_`: Regression coefficients.
- `selected_variables_`: Indices of selected variables.
- `feature_names_in_`: Feature names seen during fitting.
- `cv_results_`: Cross-validation results.

---

### DataPreprocessor Class

#### Overview

The `DataPreprocessor` class handles data scaling and preprocessing. It supports different scaling methods and ensures that the data is appropriately transformed before modeling.

#### Initialization Parameters

- `method` (str, default='standard'): Scaling method to use.
- `**kwargs`: Additional arguments specific to the scaling method.

#### Key Methods

- `fit(X)`: Fits the scaler to the data.
- `transform(X)`: Transforms the data using the fitted scaler.
- `fit_transform(X)`: Fits the scaler and transforms the data.
- `inverse_transform(X_scaled)`: Reverts the data to its original scale.

---

## Mathematical Background

### Partial Least Squares (PLS)

PLS regression seeks to find latent components that capture the maximum covariance between predictors (`X`) and responses (`Y`). It projects both `X` and `Y` into a lower-dimensional space.

### Introducing Sparsity

Sparse PLS incorporates sparsity by applying an $\ell_1$-norm penalty on the weight vectors, encouraging many coefficients to be zero. This results in variable selection and enhances interpretability.

### Optimization Problem

The optimization problem in Sparse PLS can be formulated as:

$$
\max_{\mathbf{w}, \mathbf{c}} \ \mathbf{w}^\top \mathbf{X}^\top \mathbf{Y} \mathbf{c} - \alpha (\|\mathbf{w}\|_1 + \|\mathbf{c}\|_1)
$$

Subject to:

$$
\|\mathbf{w}\|_2 = 1, \quad \|\mathbf{c}\|_2 = 1
$$

Where:

- $\mathbf{w}$ and $\mathbf{c}$ are weight vectors.
- $\alpha$ controls the level of sparsity.
- $\|\cdot\|_1$ is the $\ell_1$-norm.
- $\|\cdot\|_2$ is the $\ell_2$-norm.

---

## Development & Contributing

### Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for detailed instructions on:
- Setting up your development environment
- Code style and standards
- Running tests
- Submitting pull requests

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov matplotlib

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=sparse_pls --cov-report=html
```

### Release Process

For maintainers: See [RELEASE_GUIDE.md](RELEASE_GUIDE.md) for detailed release instructions.

**Quick release:**
```bash
./release.sh patch  # For bug fixes (0.1.2 → 0.1.3)
./release.sh minor  # For new features (0.1.2 → 0.2.0)
./release.sh major  # For breaking changes (0.1.2 → 1.0.0)
```

The release script automatically:
- Updates version numbers
- Updates CHANGELOG.md
- Creates a git tag
- Triggers GitHub Actions to publish to PyPI

---

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software in accordance with the license terms.

---

## Contact Information

If you have any questions, suggestions, or issues, please feel free to open an issue on the GitHub repository or contact the project maintainers.

---

## Author and Acknowledgments

Author and Owner of this repository : Younes AJEDDIG, Ph.D in Process & Chemical Engineering

Thanks to ChatGPT o1-preview, it's nice to have someone that do things you do not like to do even if you need it.

---

## Additional Resources

- **Scikit-learn Documentation**: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- **NumPy Documentation**: [https://numpy.org/doc/](https://numpy.org/doc/)
- **Pandas Documentation**: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
- **SciPy Documentation**: [https://docs.scipy.org/doc/](https://docs.scipy.org/doc/)

---

**Note**: Ensure that the `preprocessing.py` file containing the `DataPreprocessor` class is included in your project directory. This file should define the class and methods as used in the `SparsePLS` class.

---

**Disclaimer**: This README provides a high-level overview of the project. For detailed information, please refer to the code documentation and docstrings within the codebase.