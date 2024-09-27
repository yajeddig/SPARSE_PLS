# **SparsePLS and DataPreprocessor Documentation**

---

## **SparsePLS Class**

### **Overview**

The **Sparse Partial Least Squares (Sparse PLS) Regression** model is a dimensionality reduction technique that incorporates sparsity (variable selection) into Partial Least Squares (PLS) regression. The `SparsePLS` class implements this model, allowing for simultaneous feature selection and regression, making it particularly useful for high-dimensional datasets where interpretability and feature selection are important.

### **Parameters**

- **`n_components`** : *int, default=2*  
  Number of latent components to extract.

- **`alpha`** : *float, default=1.0*  
  Regularization parameter controlling sparsity. Higher values lead to sparser (more zero coefficients) solutions.

- **`max_iter`** : *int, default=500*  
  Maximum number of iterations in the iterative algorithm for each component.

- **`tol`** : *float, default=1e-6*  
  Tolerance for the convergence criterion. The algorithm stops when the change in weights is less than `tol`.

- **`scale`** : *bool, default=True*  
  If `True`, the data will be scaled before fitting.

- **`scale_method`** : *str, default='standard'*  
  Method used for scaling. Options include:
  - `'standard'`: Standardization to zero mean and unit variance.

- **`**kwargs`** : *dict*  
  Additional keyword arguments passed to the scaler.

### **Attributes**

- **`x_weights_`** : *ndarray of shape (n_features, n_components)*  
  Weights for the predictors (X). These weights define the direction of maximum covariance with the response.

- **`y_weights_`** : *ndarray of shape (n_targets, n_components)*  
  Weights for the response variables (Y).

- **`x_loadings_`** : *ndarray of shape (n_features, n_components)*  
  Loadings for X. They represent the covariance between X and the scores.

- **`y_loadings_`** : *ndarray of shape (n_targets, n_components)*  
  Loadings for Y.

- **`x_scores_`** : *ndarray of shape (n_samples, n_components)*  
  Scores for X, representing the projections onto the latent components.

- **`y_scores_`** : *ndarray of shape (n_samples, n_components)*  
  Scores for Y.

- **`coef_`** : *ndarray of shape (n_features, n_targets)*  
  Regression coefficients derived from the model.

- **`selected_variables_`** : *ndarray of shape (n_selected_variables,)*  
  Indices of the selected (non-zero) variables in the model.

- **`feature_names_in_`** : *ndarray of shape (n_features,), optional*  
  Feature names seen during fitting. Set only if input X has feature names.

- **`cv_results_`** : *pandas DataFrame*  
  Cross-validation results obtained after calling `optimize_parameters`.

### **Notes**

The Sparse PLS algorithm alternates between estimating sparse weight vectors for X and Y and updating scores and loadings. Sparsity is introduced via soft thresholding, promoting variable selection by shrinking some coefficients to zero.

---

### **Methods**

#### **`fit(X, Y)`**

Fit the Sparse PLS model to the training data.

**Parameters**

- **`X`** : *array-like of shape (n_samples, n_features)*  
  Training data matrix.

- **`Y`** : *array-like of shape (n_samples,) or (n_samples, n_targets)*  
  Target values.

**Returns**

- **`self`** : *object*  
  Fitted estimator.

**Description**

- Validates and optionally scales the input data.
- Initializes weight and score matrices.
- Iteratively computes components by maximizing covariance between X and Y with sparsity constraints.
- Computes regression coefficients and identifies selected variables.

---

#### **`transform(X)`**

Apply the dimensionality reduction learned during fitting to new data.

**Parameters**

- **`X`** : *array-like of shape (n_samples, n_features)*  
  New data to transform.

**Returns**

- **`X_scores`** : *ndarray of shape (n_samples, n_components)*  
  Transformed data (scores).

**Description**

- Projects the input data onto the latent components extracted during fitting.
- Applies the same scaling as during fitting if `scale=True`.

---

#### **`fit_transform(X, Y)`**

Fit the model to X and Y and apply the dimensionality reduction to X.

**Parameters**

- **`X`** : *array-like of shape (n_samples, n_features)*  
  Training data.

- **`Y`** : *array-like of shape (n_samples,) or (n_samples, n_targets)*  
  Target values.

**Returns**

- **`X_scores`** : *ndarray of shape (n_samples, n_components)*  
  Transformed data (scores).

**Description**

- Combines the `fit` and `transform` methods for efficiency.

---

#### **`predict(X)`**

Predict target values for new data.

**Parameters**

- **`X`** : *array-like of shape (n_samples, n_features)*  
  Samples to predict.

**Returns**

- **`Y_pred`** : *ndarray of shape (n_samples,) or (n_samples, n_targets)*  
  Predicted target values.

**Description**

- Uses the regression coefficients computed during fitting to predict targets.
- Applies scaling transformations if necessary.

---

#### **`optimize_parameters(X, Y, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=1, verbose=0, return_models=False)`**

Optimize hyperparameters using cross-validation.

**Parameters**

- **`X`** : *array-like of shape (n_samples, n_features)*  
  Training data.

- **`Y`** : *array-like of shape (n_samples,) or (n_samples, n_targets)*  
  Target values.

- **`param_grid`** : *dict*  
  Dictionary specifying parameter grid to search. Keys are parameter names, and values are lists of parameter settings to try.

- **`cv`** : *int or cross-validation generator, default=5*  
  Determines the cross-validation splitting strategy.

- **`scoring`** : *str or callable, default='neg_mean_squared_error'*  
  Metric to evaluate the performance of the model.

- **`n_jobs`** : *int, default=1*  
  Number of jobs to run in parallel. `-1` means using all processors.

- **`verbose`** : *int, default=0*  
  Controls the verbosity: higher values lead to more messages.

- **`return_models`** : *bool, default=False*  
  If `True`, stores the models trained for each parameter combination.

**Returns**

- **`self`** : *object*  
  Estimator with the best found parameters.

**Description**

- Performs grid search over specified hyperparameters using cross-validation.
- Records results in `cv_results_`.
- Refits the model on the entire dataset with the best parameters found.

---

### **Internal Methods**

#### **`_sparse_pls_component(X, Y)`**

Compute a single sparse PLS component.

**Parameters**

- **`X`** : *ndarray of shape (n_samples, n_features)*  
  Residual matrix of predictors.

- **`Y`** : *ndarray of shape (n_samples, n_targets)*  
  Residual matrix of responses.

**Returns**

- **`w`** : *ndarray of shape (n_features, 1)*  
  Weight vector for X.

- **`c`** : *ndarray of shape (n_targets, 1)*  
  Weight vector for Y.

**Description**

- Iteratively updates weight vectors for X and Y by maximizing covariance with sparsity constraints.
- Uses soft thresholding to induce sparsity in the weight vectors.

---

#### **`_soft_thresholding(z, alpha)`**

Apply soft thresholding to induce sparsity.

**Parameters**

- **`z`** : *ndarray*  
  Input vector.

- **`alpha`** : *float*  
  Thresholding parameter controlling sparsity.

**Returns**

- **`z_thresh`** : *ndarray*  
  Thresholded vector with induced sparsity.

**Description**

- Shrinks elements of `z` towards zero by `alpha`.
- Elements with absolute value less than `alpha` become zero.

---

#### **`_get_selected_variables()`**

Identify variables selected by the model (non-zero weights).

**Returns**

- **`selected_vars`** : *ndarray of shape (n_selected_variables,)*  
  Indices of selected variables.

**Description**

- Scans the weight matrix `x_weights_` to find variables with non-zero weights across components.

---

## **DataPreprocessor Class**

### **Overview**

The `DataPreprocessor` class handles data scaling and preprocessing steps required before fitting the model. It supports standard scaling methods and can be extended to include additional preprocessing techniques.

### **Parameters**

- **`method`** : *str, default='standard'*  
  Scaling method to use. Options include:
  - `'standard'`: Scales features to have zero mean and unit variance.

- **`**kwargs`** : *dict*  
  Additional arguments specific to the chosen scaling method.

### **Methods**

#### **`fit(X)`**

Fit the scaler to the data.

**Parameters**

- **`X`** : *array-like of shape (n_samples, n_features)*  
  Data to fit the scaler.

**Returns**

- **`self`** : *object*  
  Fitted preprocessor.

**Description**

- Learns scaling parameters (e.g., mean and variance) from the data.

---

#### **`transform(X)`**

Apply the scaling transformation to the data.

**Parameters**

- **`X`** : *array-like of shape (n_samples, n_features)*  
  Data to transform.

**Returns**

- **`X_scaled`** : *array-like of shape (n_samples, n_features)*  
  Scaled data.

**Description**

- Transforms the data using the parameters learned during `fit`.

---

#### **`fit_transform(X)`**

Fit the scaler to the data and then transform it.

**Parameters**

- **`X`** : *array-like of shape (n_samples, n_features)*  
  Data to fit and transform.

**Returns**

- **`X_scaled`** : *array-like of shape (n_samples, n_features)*  
  Scaled data.

**Description**

- Convenience method combining `fit` and `transform`.

---

#### **`inverse_transform(X_scaled)`**

Undo the scaling transformation.

**Parameters**

- **`X_scaled`** : *array-like of shape (n_samples, n_features)*  
  Scaled data to inverse transform.

**Returns**

- **`X_original`** : *array-like of shape (n_samples, n_features)*  
  Data restored to original scale.

**Description**

- Reverts the data back to its original scale using the parameters learned during fitting.

---

### **Usage Example**

```python
from model import SparsePLS
from preprocessing import DataPreprocessor

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
model.optimize_parameters(X_train, y_train, param_grid=param_grid)

# Access selected variables
selected_features = model.feature_names_in_[model.selected_variables_]
```

---

### **Key Points**

- **SparsePLS** combines dimensionality reduction and feature selection, making it suitable for high-dimensional data.
- **DataPreprocessor** ensures that data is appropriately scaled, which is crucial for algorithms like PLS that are sensitive to the scale of the variables.
- The `optimize_parameters` method allows for automated hyperparameter tuning using cross-validation, aiding in model selection.
- The model's attributes provide access to the internal components and results, enabling in-depth analysis and interpretation.

---

## **Conclusion**

The `SparsePLS` and `DataPreprocessor` classes provide a robust framework for performing sparse partial least squares regression with built-in support for scaling and hyperparameter optimization. By leveraging sparsity, the model enhances interpretability and focuses on the most relevant features, making it a powerful tool for regression tasks in high-dimensional settings.

For more detailed examples and use cases, refer to the respective method documentation and consider exploring real-world datasets to fully appreciate the capabilities of these classes.

---

**Note:** Ensure that all dependencies such as `numpy`, `pandas`, `scikit-learn`, and any custom modules like `preprocessing` are properly installed and imported in your working environment.