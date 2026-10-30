"""Unfitted feature transformation to place inside each model pipeline."""

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor() -> ColumnTransformer:
    """Return a fresh transformer; fit it only on a training fold.

    German Credit has no missing values in its documented version. Imputers
    also make inference behavior explicit for future inputs with missing data.
    Numeric features keep their original scale for Random Forest.
    """
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical,
                make_column_selector(dtype_include=["object", "category", "string"]),
            ),
            ("numeric", numeric, make_column_selector(dtype_include="number")),
        ],
        remainder="drop",
    )
