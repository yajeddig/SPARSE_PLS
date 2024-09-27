import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.model_selection import ParameterGrid
from scipy.linalg import pinv
from preprocessing import DataPreprocessor
from joblib import Parallel, delayed
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
from sklearn.base import is_classifier


class SparsePLS(BaseEstimator, RegressorMixin):
    """
    Sparse Partial Least Squares (Sparse PLS) Regression.

    The `SparsePLS` class implements a Sparse Partial Least Squares regression model,
    which is a dimensionality reduction technique that incorporates sparsity (variable selection)
    into the Partial Least Squares (PLS) regression.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.

    alpha : float, default=1.0
        Regularization parameter controlling sparsity. Higher values lead to sparser solutions.

    max_iter : int, default=500
        Maximum number of iterations in the iterative algorithm.

    tol : float, default=1e-6
        Tolerance for the stopping condition.

    scale : bool, default=True
        Whether to scale X and Y.

    scale_method : str, default='standard'
        Method used for scaling. Options: 'standard'.

    **kwargs : dict
        Additional keyword arguments for the scaler.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        Weights for X.

    y_weights_ : ndarray of shape (n_targets, n_components)
        Weights for Y.

    x_loadings_ : ndarray of shape (n_features, n_components)
        Loadings for X.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        Loadings for Y.

    x_scores_ : ndarray of shape (n_samples, n_components)
        Scores for X.

    y_scores_ : ndarray of shape (n_samples, n_components)
        Scores for Y.

    coef_ : ndarray of shape (n_features, n_targets)
        Coefficients of the regression model.

    selected_variables_ : ndarray of shape (n_selected_variables,)
        Indices of the selected variables.

    feature_names_in_ : ndarray of shape (n_features,), optional
        Feature names seen during fit. Defined only when X has feature names that are all strings.

    cv_results_ : pandas DataFrame
        Cross-validation results, available after calling `optimize_parameters`.

    Notes
    -----
    The algorithm is based on an iterative procedure that alternates between estimating
    sparse weight vectors for X and Y, and updating scores and loadings.
    """

    def __init__(self, n_components=2, alpha=1.0, max_iter=500, tol=1e-6,
                 scale=True, scale_method='standard', **kwargs):
        # Initialize model parameters
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale
        self.scale_method = scale_method
        self.scaler_kwargs = kwargs  # Additional arguments for the scaler

    def fit(self, X, Y):
        """
        Fit the Sparse PLS model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate input data and set feature_names_in_
        X, Y = self._validate_data(
            X, Y,
            accept_sparse=False,
            dtype=None,
            force_all_finite=True,
            multi_output=True,
            y_numeric=True,
            reset=True
        )

        # Debug statement to check if feature_names_in_ is set
        if hasattr(self, 'feature_names_in_'):
            print("feature_names_in_ is set during fit.")
        else:
            print("feature_names_in_ is NOT set during fit.")

        n_samples, n_features = X.shape
        n_targets = Y.shape[1] if Y.ndim > 1 else 1

        # Preprocessing
        if self.scale:
            # Initialize scalers for X and Y
            self._x_scaler = DataPreprocessor(method=self.scale_method, **self.scaler_kwargs)
            self._y_scaler = DataPreprocessor(method=self.scale_method, **self.scaler_kwargs)
            # Fit and transform X
            X = self._x_scaler.fit_transform(X)
            # Reshape Y if necessary and fit and transform Y
            Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
            Y = self._y_scaler.fit_transform(Y)
        else:
            self._x_scaler = None
            self._y_scaler = None

        # Initialize matrices to store results
        self.x_weights_ = np.zeros((n_features, self.n_components))
        self.y_weights_ = np.zeros((n_targets, self.n_components))
        self.x_scores_ = np.zeros((n_samples, self.n_components))
        self.y_scores_ = np.zeros((n_samples, self.n_components))
        self.x_loadings_ = np.zeros((n_features, self.n_components))
        self.y_loadings_ = np.zeros((n_targets, self.n_components))

        # Residual matrices for deflation
        X_residual = X.copy()
        Y_residual = Y.copy()

        # Iterate over the number of components
        for k in range(self.n_components):
            # Compute one sparse PLS component
            w, c = self._sparse_pls_component(X_residual, Y_residual)

            # Compute scores
            t = X_residual @ w
            u = Y_residual @ c

            # Normalize scores to avoid numerical issues
            t_norm = np.linalg.norm(t)
            if t_norm == 0:
                # If the norm is zero, break the loop
                break
            t /= t_norm
            u /= t_norm

            # Compute loadings
            p = X_residual.T @ t
            q = Y_residual.T @ t

            # Store results
            self.x_weights_[:, k] = w.ravel()
            self.y_weights_[:, k] = c.ravel()
            self.x_scores_[:, k] = t.ravel()
            self.y_scores_[:, k] = u.ravel()
            self.x_loadings_[:, k] = p.ravel()
            self.y_loadings_[:, k] = q.ravel()

            # Deflate data matrices
            X_residual -= np.outer(t, p)
            Y_residual -= np.outer(t, q)

        # Compute regression coefficients
        # Using pseudo-inverse for stability
        self.coef_ = self.x_weights_ @ pinv(self.x_loadings_.T @ self.x_weights_) @ self.y_loadings_.T

        # Select variables based on non-zero weights
        self.selected_variables_ = self._get_selected_variables()

        return self

    def _sparse_pls_component(self, X, Y):
        """
        Compute one sparse PLS component.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Residual matrix of X.

        Y : ndarray of shape (n_samples, n_targets)
            Residual matrix of Y.

        Returns
        -------
        w : ndarray of shape (n_features, 1)
            Weight vector for X.

        c : ndarray of shape (n_targets, 1)
            Weight vector for Y.
        """
        n_features = X.shape[1]
        n_targets = Y.shape[1] if Y.ndim > 1 else 1

        # Initialize weight vector for Y with random values and normalize
        c = np.random.rand(n_targets, 1)
        c /= np.linalg.norm(c)

        for iteration in range(self.max_iter):
            # Update weight vector w for X
            z_w = X.T @ Y @ c  # Compute gradient
            w = self._soft_thresholding(z_w, self.alpha)  # Apply soft thresholding
            if np.linalg.norm(w) == 0:
                # If all weights are zero, stop the iteration
                break
            w /= np.linalg.norm(w)  # Normalize weights

            # Update score vector t for X
            t = X @ w

            # Update weight vector c for Y
            z_c = Y.T @ t  # Compute gradient
            c_new = self._soft_thresholding(z_c, self.alpha)  # Apply soft thresholding
            if np.linalg.norm(c_new) == 0:
                # If all weights are zero, stop the iteration
                break
            c_new /= np.linalg.norm(c_new)  # Normalize weights

            # Check convergence
            if np.linalg.norm(c_new - c) < self.tol:
                c = c_new
                break

            c = c_new  # Update c for next iteration

        return w, c

    def _soft_thresholding(self, z, alpha):
        """
        Apply soft thresholding to vector z.

        Parameters
        ----------
        z : ndarray
            Input vector.

        alpha : float
            Thresholding parameter.

        Returns
        -------
        z_thresh : ndarray
            Thresholded vector.
        """
        # Apply soft thresholding element-wise
        return np.sign(z) * np.maximum(np.abs(z) - alpha, 0)

    def _get_selected_variables(self):
        """
        Identify the indices of variables with non-zero weights.

        Returns
        -------
        selected_vars : ndarray of shape (n_selected_variables,)
            Indices of the selected variables.
        """
        # Selected variables are those with non-zero weights in x_weights_
        non_zero_weights = np.any(self.x_weights_ != 0, axis=1)
        selected_vars = np.where(non_zero_weights)[0]
        return selected_vars

    def transform(self, X):
        """
        Apply the dimensionality reduction learned on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_scores : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = check_array(X)
        # Apply scaling if necessary
        if self.scale and self._x_scaler is not None:
            X = self._x_scaler.transform(X)
        # Project data onto the components
        X_scores = X @ self.x_weights_
        return X_scores

    def fit_transform(self, X, Y):
        """
        Fit the model to X and Y and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        X_scores : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        # Fit the model
        self.fit(X, Y)
        # Return the scores for X
        return self.x_scores_

    def predict(self, X):
        """
        Predict target values for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        Y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted target values.
        """
        check_is_fitted(self)
        X = check_array(X)
        # Apply scaling if necessary
        if self.scale and self._x_scaler is not None:
            X = self._x_scaler.transform(X)
        # Predict target values
        Y_pred = X @ self.coef_
        # Apply inverse scaling if necessary
        if self.scale and self._y_scaler is not None:
            Y_pred = Y_pred.reshape(-1, 1) if Y_pred.ndim == 1 else Y_pred
            Y_pred = self._y_scaler.inverse_transform(Y_pred)
            Y_pred = Y_pred.ravel() if Y_pred.shape[1] == 1 else Y_pred
        return Y_pred

    def optimize_parameters(self, X, Y, param_grid, cv=5, scoring='neg_mean_squared_error',
                            n_jobs=1, verbose=0, return_models=False):
        """
        Optimize hyperparameters using cross-validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        param_grid : dict
            Dictionary with parameter names (`str`) as keys and lists of parameter settings to try as values.

        cv : int, cross-validation generator or an iterable, default=5
            Determines the cross-validation splitting strategy.

        scoring : str, callable or None, default='neg_mean_squared_error'
            A string (see model evaluation documentation) or a scorer callable object/function with signature
            ``scorer(estimator, X, y)``.

        n_jobs : int, default=1
            Number of jobs to run in parallel. ``-1`` means using all processors.

        verbose : int, default=0
            Controls the verbosity: the higher, the more messages.

        return_models : bool, default=False
            Whether to store the models trained for each parameter combination.

        Returns
        -------
        self : object
            Returns the instance itself with the best parameters found.

        Notes
        -----
        The best parameters are stored in the instance attributes and the model is refitted on the entire dataset.
        """
        # Validate input data
        X, Y = self._validate_data(X, Y, multi_output=True, y_numeric=True, reset=False)
        cv = check_cv(cv=cv, y=Y, classifier=is_classifier(self))
        scorer = check_scoring(self, scoring=scoring)

        param_list = list(ParameterGrid(param_grid))
        results = []

        if verbose > 0:
            print("Optimizing over parameter grid:")
            for params in param_list:
                print(params)

        def fit_and_score(params, train_idx, test_idx):
            """
            Fit the model with given parameters and compute the score on the test set.

            Parameters
            ----------
            params : dict
                Parameter settings to try.

            train_idx : array-like
                Indices for the training set.

            test_idx : array-like
                Indices for the test set.

            Returns
            -------
            score : float
                Computed score on the test set.

            model : object, optional
                Trained model instance if `return_models` is True.
            """
            # Use .iloc if X and Y are DataFrames/Series
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
            else:
                X_train = X[train_idx]
                X_test = X[test_idx]

            if isinstance(Y, (pd.Series, pd.DataFrame)):
                Y_train = Y.iloc[train_idx]
                Y_test = Y.iloc[test_idx]
            else:
                Y_train = Y[train_idx]
                Y_test = Y[test_idx]

            # Clone the model with current parameters
            model = clone(self)
            model.set_params(**params)
            # Fit the model on the training data
            model.fit(X_train, Y_train)
            # Predict on the test data
            Y_pred = model.predict(X_test)
            # Compute the score
            score = scorer(model, X_test, Y_test)

            return score, model if return_models else None

        # Iterate over all parameter combinations
        for params in param_list:
            splits = list(cv.split(X, Y))

            # Parallel computation of fit and score for each fold
            parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
            scores_models = parallel(
                delayed(fit_and_score)(params, train_idx, test_idx)
                for train_idx, test_idx in splits
            )

            # Extract scores and models
            scores = [sm[0] for sm in scores_models]
            models = [sm[1] for sm in scores_models] if return_models else None

            # Compute mean and standard deviation of scores
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            result = {
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'scores': scores
            }
            if return_models:
                result['models'] = models

            results.append(result)

            if verbose > 0:
                print(f"Params: {params}, Mean Score: {mean_score:.4f}, Std: {std_score:.4f}")

        # Convert results to DataFrame for easy access
        self.cv_results_ = pd.DataFrame(results)

        # Select best parameters (maximize the score)
        best_result = max(results, key=lambda x: x['mean_score'])
        best_params = best_result['params']
        self.set_params(**best_params)

        if verbose > 0:
            print(f"Best parameters found: {best_params}")

        # Refit on the entire dataset with the best parameters
        # Ensure X is a DataFrame with column names
        if not isinstance(X, pd.DataFrame):
            # Assuming you have access to the original feature names
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        elif X.columns is None or not list(X.columns):
            # If columns are missing, set them
            X.columns = self.feature_names_in_

        self.fit(X, Y)
        return self
