"""
Machine Learning Utilities

Author: Sakshi Raut
Project: Geldium Credit Delinquency Risk Analysis

Description:
Reusable utilities for building, training,
evaluating, visualizing and saving
machine learning models.
"""

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

def build_logistic_regression(random_state=42):
    """
    Create Logistic Regression model.
    """

    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000
    )

    return model

def train_model(model, X_train, y_train):
    """
    Train machine learning model.
    """

    model.fit(X_train, y_train)

    return model

def predict(model, X_test):
    """
    Generate predictions.
    """

    return model.predict(X_test)

def predict_probability(model, X_test):
    """
    Generate prediction probabilities.
    """

    return model.predict_proba(X_test)[:, 1]

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    """

    y_pred = predict(model, X_test)

    y_prob = predict_probability(model, X_test)

    metrics = {

        "Accuracy":
            accuracy_score(y_test, y_pred),

        "Precision":
            precision_score(y_test, y_pred),

        "Recall":
            recall_score(y_test, y_pred),

        "F1 Score":
            f1_score(y_test, y_pred),

        "ROC AUC":
            roc_auc_score(y_test, y_prob)

    }

    return pd.DataFrame(
        metrics.items(),
        columns=["Metric", "Value"]
    )

def generate_classification_report(
    model,
    X_test,
    y_test
):
    """
    Generate classification report.
    """

    y_pred = predict(model, X_test)

    report = classification_report(
        y_test,
        y_pred
    )

    print(report)

def plot_confusion_matrix(
    model,
    X_test,
    y_test
):
    """
    Plot confusion matrix.
    """

    y_pred = predict(model, X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.show()

def plot_roc_curve(
    model,
    X_test,
    y_test
):
    """
    Plot ROC Curve.
    """

    probabilities = predict_probability(
        model,
        X_test
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    plt.figure(figsize=(6,5))

    plt.plot(
        fpr,
        tpr,
        label="ROC Curve"
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.show()

def logistic_feature_importance(
    model,
    feature_names
):
    """
    Display logistic regression coefficients.
    """

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Coefficient": model.coef_[0]

    })

    importance["Absolute"] = importance[
        "Coefficient"
    ].abs()

    importance = importance.sort_values(
        by="Absolute",
        ascending=False
    )

    return importance

def save_model(
    model,
    filepath
):
    """
    Save trained model.
    """

    joblib.dump(
        model,
        filepath
    )

    print(
        f"Model saved to {filepath}"
    )

def load_model(filepath):
    """
    Load trained model.
    """

    return joblib.load(filepath)

def evaluate_and_visualize(
    model,
    X_test,
    y_test
):
    """
    Run full evaluation workflow.
    """

    print("=" * 60)
    print("Model Performance")
    print("=" * 60)

    display(evaluate_model(model, X_test, y_test))

    print("\nClassification Report\n")
    generate_classification_report(
        model,
        X_test,
        y_test
    )

    plot_confusion_matrix(
        model,
        X_test,
        y_test
    )

    plot_roc_curve(
        model,
        X_test,
        y_test
    )