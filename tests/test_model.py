import sys
import os

# Ajouter le répertoire parent au chemin de recherche
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from numpy.testing import assert_array_almost_equal
from model import SparsePLS

@pytest.fixture
def data():
    X_np, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    feature_names = [f'feature_{i}' for i in range(X_np.shape[1])]
    X_df = pd.DataFrame(X_np, columns=feature_names)
    return X_np, X_df, y, feature_names

def test_numpy_input(data):
    X_np, _, y, _ = data
    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X_np, y)
    assert hasattr(model, 'coef_'), "Le modèle n'a pas été entraîné correctement avec numpy.ndarray"

def test_pandas_input(data):
    _, X_df, y, _ = data
    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X_df, y)
    assert hasattr(model, 'coef_'), "Le modèle n'a pas été entraîné correctement avec pandas.DataFrame"

def test_column_names_preserved(data):
    _, X_df, y, feature_names = data
    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X_df, y)
    assert model.feature_names_in_ is not None, "Les noms de colonnes n'ont pas été conservés"
    assert list(model.feature_names_in_) == feature_names, "Les noms de colonnes ne correspondent pas"

def test_prediction(data):
    X_np, X_df, y, feature_names = data
    X_test_np, _ = make_regression(n_samples=10, n_features=10, noise=0.1, random_state=24)
    X_test_df = pd.DataFrame(X_test_np, columns=feature_names)

    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X_df, y)

    y_pred_np = model.predict(X_test_np)
    y_pred_df = model.predict(X_test_df)

    assert_array_almost_equal(y_pred_np, y_pred_df, err_msg="Les prédictions diffèrent entre numpy.ndarray et pandas.DataFrame")

def test_invalid_input():
    X_invalid = "invalid_input"
    y_invalid = [1, 2, 3]

    model = SparsePLS(n_components=2, alpha=0.1)

    with pytest.raises(ValueError):
        model.fit(X_invalid, y_invalid)
