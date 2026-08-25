"""
ML Evaluation Utilities — Production-Grade Model Assessment

Implements the ML Engineer role checklist:
- Cross-validation for model stability
- Bootstrapped confidence intervals
- Error analysis helpers

Usage:
    from ml_evaluation_utils import evaluate_classifier, evaluate_regressor
    
    results = evaluate_classifier(model, X_test, y_test, model_name="XGBoost")
    print(results)
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, Any, Optional


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Compute bootstrapped confidence interval for any metric.
    
    This implements the ML Engineer checklist requirement:
    "Compute bootstrapped confidence intervals with 1000 resamples.
    Report the 95% CI alongside each point estimate."
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        metric_fn: Function that takes (y_true, y_pred) and returns a score
        n_bootstrap: Number of bootstrap resamples (default: 1000)
        confidence: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        Dict with 'point_estimate', 'ci_lower', 'ci_upper'
    """
    scores = []
    n = len(y_true)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        score = metric_fn(y_true[indices], y_pred[indices])
        scores.append(score)
    
    scores = np.array(scores)
    alpha = 1 - confidence
    ci_lower = np.percentile(scores, 100 * alpha / 2)
    ci_upper = np.percentile(scores, 100 * (1 - alpha / 2))
    
    return {
        "point_estimate": float(metric_fn(y_true, y_pred)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence_level": confidence,
        "n_bootstrap": n_bootstrap
    }


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Comprehensive classifier evaluation with cross-validation and confidence intervals.
    
    Implements ML Engineer checklist:
    - Metric selection (accuracy, F1, precision, recall)
    - Confidence intervals (bootstrapped)
    - Cross-validation for stability
    - Confusion matrix
    - Error analysis
    """
    y_pred = model.predict(X_test)
    
    # Basic metrics
    results = {
        "model_name": model_name,
        "test_size": len(y_test),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average='macro')),
        "f1_weighted": float(f1_score(y_test, y_pred, average='weighted')),
        "precision_macro": float(precision_score(y_test, y_pred, average='macro')),
        "recall_macro": float(recall_score(y_test, y_pred, average='macro')),
    }
    
    # Cross-validation (if model supports it)
    try:
        cv_scores = cross_val_score(model, X_test, y_test, cv=cv_folds, scoring='accuracy')
        results["cv_accuracy_mean"] = float(cv_scores.mean())
        results["cv_accuracy_std"] = float(cv_scores.std())
        results["cv_folds"] = cv_folds
    except Exception:
        results["cv_accuracy_mean"] = None
        results["cv_accuracy_std"] = None
    
    # Bootstrapped confidence intervals for key metrics
    results["confidence_intervals"] = {
        "accuracy": bootstrap_confidence_interval(y_test, y_pred, accuracy_score),
        "f1": bootstrap_confidence_interval(y_test, y_pred, lambda y, p: f1_score(y, p, average='macro')),
    }
    
    # Confusion matrix
    results["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    
    # Classification report
    results["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
    
    # Error analysis
    misclassified = np.where(y_pred != y_test)[0]
    results["error_analysis"] = {
        "n_misclassified": len(misclassified),
        "misclassification_rate": float(len(misclassified) / len(y_test)),
        "misclassified_indices": misclassified.tolist()[:20]  # First 20
    }
    
    return results


def evaluate_regressor(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Comprehensive regressor evaluation with cross-validation and confidence intervals.
    
    Implements ML Engineer checklist:
    - RMSE, MAE, R² metrics
    - Cross-validation for stability
    - Confidence intervals
    """
    y_pred = model.predict(X_test)
    
    # Basic metrics
    results = {
        "model_name": model_name,
        "test_size": len(y_test),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    
    # Cross-validation
    try:
        cv_scores = cross_val_score(model, X_test, y_test, cv=cv_folds, scoring='r2')
        results["cv_r2_mean"] = float(cv_scores.mean())
        results["cv_r2_std"] = float(cv_scores.std())
        results["cv_folds"] = cv_folds
    except Exception:
        results["cv_r2_mean"] = None
        results["cv_r2_std"] = None
    
    # Confidence intervals for R²
    results["confidence_intervals"] = {
        "r2": bootstrap_confidence_interval(y_test, y_pred, r2_score)
    }
    
    return results


# Example usage (for documentation)
if __name__ == "__main__":
    print(__doc__)
