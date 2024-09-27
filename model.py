import numpy as np
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from sklearn.model_selection import KFold
from scipy.linalg import pinv
from preprocessing import DataPreprocessor


class SparsePLS(BaseEstimator, TransformerMixin, RegressorMixin):
    """
    Sparse Partial Least Squares (Sparse PLS) Regression.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.

    alpha : float, default=1.0
        Regularization parameter controlling sparsity.

    max_iter : int, default=500
        Maximum number of iterations in the iterative algorithm.

    tol : float, default=1e-06
        Tolerance for the stopping condition.

    scale : bool, default=True
        Whether to scale X and Y.

    scale_method : str, default='standard'
        Method used for scaling. Options: 'standard', 'robust', 'quantile_uniform', 'quantile_normal'.

    **kwargs : dict
        Additional keyword arguments for the scaler.
    """

    def __init__(self, n_components=2, alpha=1.0, max_iter=500, tol=1e-6,
                 scale=True, scale_method='standard', **kwargs):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale
        self.scale_method = scale_method
        self.scaler_kwargs = kwargs  # Additional arguments for the scaler

    def fit(self, X, Y):
        # Conserver les noms de colonnes si X est un DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns
        else:
            self.feature_names_in_ = None

        # Validation des données
        X, Y = check_X_y(X, Y, multi_output=True, y_numeric=True)

        n_samples, n_features = X.shape
        n_targets = Y.shape[1] if Y.ndim > 1 else 1
    
        # Prétraitement
        if self.scale:
            self._x_scaler = DataPreprocessor(method=self.scale_method, **self.scaler_kwargs)
            self._y_scaler = DataPreprocessor(method=self.scale_method, **self.scaler_kwargs)
            X = self._x_scaler.fit_transform(X)
            Y = Y.reshape(-1, 1)
            Y = self._y_scaler.fit_transform(Y)
        else:
            self._x_scaler = None
            self._y_scaler = None

        # Initialisation des matrices pour stocker les résultats
        self.x_weights_ = np.zeros((n_features, self.n_components))
        self.y_weights_ = np.zeros((n_targets, self.n_components))
        self.x_scores_ = np.zeros((n_samples, self.n_components))
        self.y_scores_ = np.zeros((n_samples, self.n_components))
        self.x_loadings_ = np.zeros((n_features, self.n_components))
        self.y_loadings_ = np.zeros((n_targets, self.n_components))

        # Matrices résiduelles
        X_residual = X.copy()
        Y_residual = Y.copy()

        for k in range(self.n_components):
            # Calcul de la composante sparse PLS
            w, c = self._sparse_pls_component(X_residual, Y_residual)
            t = X_residual @ w
            u = Y_residual @ c

            # Normalisation des scores pour éviter les problèmes numériques
            t_norm = np.linalg.norm(t)
            if t_norm == 0:
                break
            t /= t_norm
            u /= t_norm

            # Calcul des charges
            p = X_residual.T @ t
            q = Y_residual.T @ t

            # Stockage des résultats
            self.x_weights_[:, k] = w.ravel()
            self.y_weights_[:, k] = c.ravel()
            self.x_scores_[:, k] = t.ravel()
            self.y_scores_[:, k] = u.ravel()
            self.x_loadings_[:, k] = p.ravel()
            self.y_loadings_[:, k] = q.ravel()

            # Déflation des données
            X_residual -= np.outer(t, p)
            Y_residual -= np.outer(t, q)

        # Calcul des coefficients de régression
        self.coef_ = self.x_weights_ @ pinv(self.x_loadings_.T @ self.x_weights_) @ self.y_loadings_.T

        # Sélection des variables
        self.selected_variables_ = self._get_selected_variables()

        return self

    
    def _sparse_pls_component(self, X, Y):
        """
        Compute one sparse PLS component.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Residual matrix of X.

        Y : array-like of shape (n_samples, n_targets)
            Residual matrix of Y.

        Returns
        -------
        w : array-like of shape (n_features, 1)
            Weight vector for X.

        c : array-like of shape (n_targets, 1)
            Weight vector for Y.
        """
        n_features = X.shape[1]
        n_targets = Y.shape[1] if Y.ndim > 1 else 1

        # Initialize weight vector for Y
        c = np.random.rand(n_targets, 1)
        c /= np.linalg.norm(c)

        for iteration in range(self.max_iter):
            # Update w
            z_w = X.T @ Y @ c
            w = self._soft_thresholding(z_w, self.alpha)
            if np.linalg.norm(w) == 0:
                break
            w /= np.linalg.norm(w)

            # Update t
            t = X @ w

            # Update c
            z_c = Y.T @ t
            c_new = self._soft_thresholding(z_c, self.alpha)
            if np.linalg.norm(c_new) == 0:
                break
            c_new /= np.linalg.norm(c_new)

            # Check convergence
            if np.linalg.norm(c_new - c) < self.tol:
                c = c_new
                break

            c = c_new

        return w, c

    def _soft_thresholding(self, z, alpha):
        """
        Apply soft thresholding to vector z.

        Parameters
        ----------
        z : array-like
            Input vector.

        alpha : float
            Thresholding parameter.

        Returns
        -------
        z_thresh : array-like
            Thresholded vector.
        """
        return np.sign(z) * np.maximum(np.abs(z) - alpha, 0)

    def _get_selected_variables(self):
        """
        Identify the indices of variables with non-zero weights.

        Returns
        -------
        selected_vars : array-like of shape (n_selected_variables,)
            Indices of the selected variables.
        """
        # Les variables sélectionnées sont celles dont les poids dans x_weights_ ne sont pas nuls
        non_zero_weights = np.any(self.x_weights_ != 0, axis=1)
        selected_vars = np.where(non_zero_weights)[0]
        return selected_vars
    
    def transform(self, X):
        """
        Apply the dimension reduction learned on the train data.
        """
        check_is_fitted(self)
        X = check_array(X)
        if self.scale and self._x_scaler is not None:
            X = self._x_scaler.transform(X)
        X_scores = X @ self.x_weights_
        return X_scores

    def fit_transform(self, X, Y):
        """
        Fit the model to X and Y and apply the dimensionality reduction on X.
        """
        self.fit(X, Y)
        return self.x_scores_

    def predict(self, X):
        """
        Predict target values for X.
        """
        check_is_fitted(self)
        X = check_array(X)
        if self.scale and self._x_scaler is not None:
            X = self._x_scaler.transform(X)
        Y_pred = X @ self.coef_
        if self.scale and self._y_scaler is not None:
            Y_pred = self._y_scaler.inverse_transform(Y_pred)
        return Y_pred

    def optimize_parameters(self, X, Y, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=1, verbose=0, return_models=False):
        """
        Optimize hyperparameters using cross-validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        param_grid : dict
            Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.

        cv : int, default=5
            Number of folds in K-Fold cross-validation.

        scoring : str, callable or None, default='neg_mean_squared_error'
            Scoring method to use.

        n_jobs : int, default=1
            Number of jobs to run in parallel.

        verbose : int, default=0
            Verbosity level.

        return_models : bool, default=False
            Whether to store the models trained for each parameter combination.

        Returns
        -------
        self : object
            Returns the instance itself with the best parameters found.
        """
        from joblib import Parallel, delayed
        from sklearn.base import is_classifier
        from sklearn.model_selection import check_cv
        from sklearn.metrics import check_scoring

        X, Y = check_X_y(X, Y, multi_output=True, y_numeric=True)
        cv = check_cv(cv=cv, y=Y, classifier=is_classifier(self))
        scorer = check_scoring(self, scoring=scoring)

        param_list = list(ParameterGrid(param_grid))
        results = []

        if verbose > 0:
            print("Optimizing over parameter grid:")
            for params in param_list:
                print(params)

        def fit_and_score(params, train_idx, test_idx):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            model = clone(self)
            model.set_params(**params)
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            score = scorer(model, X_test, Y_test)

            return score, model if return_models else None

        for params in param_list:
            splits = list(cv.split(X, Y))

            parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
            scores_models = parallel(
                delayed(fit_and_score)(params, train_idx, test_idx)
                for train_idx, test_idx in splits
            )

            scores = [sm[0] for sm in scores_models]
            models = [sm[1] for sm in scores_models] if return_models else None

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

        # Convertir les résultats en DataFrame pour faciliter le traçage
        self.cv_results_ = pd.DataFrame(results)

        # Sélection des meilleurs paramètres (maximiser le score)
        best_result = max(results, key=lambda x: x['mean_score'])
        best_params = best_result['params']
        self.set_params(**best_params)

        if verbose > 0:
            print(f"Best parameters found: {best_params}")

        # Refit sur l'ensemble des données avec les meilleurs paramètres
        self.fit(X, Y)

        return self
