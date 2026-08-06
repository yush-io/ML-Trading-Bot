from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_baseline_model(X_train, y_train):
    """Train the first baseline classification model.

    The model predicts whether the next trading day is more likely to be up.
    Logistic regression is simple and explainable, which makes it a good first
    benchmark before trying more complex models.
    """

    # A Pipeline chains preprocessing and modeling into one object.
    model = Pipeline(
        steps=[
            # Scale features so values with larger ranges do not dominate.
            ("scaler", StandardScaler()),
            # Logistic regression outputs probabilities for binary classes.
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    # Fit means "learn the relationship between X_train and y_train".
    model.fit(X_train, y_train)
    return model


def classification_metrics(y_true, predicted_labels, predicted_probabilities) -> dict[str, float]:
    """Calculate basic ML diagnostics for the classifier.

    These metrics help us understand prediction quality, but trading performance
    is judged later with risk metrics from the backtest.
    """

    return {
        "accuracy": accuracy_score(y_true, predicted_labels),
        "precision": precision_score(y_true, predicted_labels, zero_division=0),
        "roc_auc": roc_auc_score(y_true, predicted_probabilities),
    }
