"""
The module contains functions to calculate, visualise and log 
model metrics in a fraudulent transaction detection system.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union, Any
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine

# Configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definition of types
MetricsDict = Dict[str, float]


def calculate_classification_metrics(y_true: np.ndarray, 
                                    y_pred: np.ndarray,
                                    y_prob: Optional[np.ndarray] = None) -> MetricsDict:
    """
    Calculates the basic metrics of binary classification.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        y_prob: Probabilities of belonging to a positive class (optional)
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # If probabilities are provided, calculate the AUC-ROC
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['average_precision'] = average_precision_score(y_true, y_prob)
    
    logger.info(f"Classification metrics calculated: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"f1={metrics['f1']:.4f}")
    
    return metrics


def calculate_cost_sensitive_metrics(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    cost_fn: float = 1.0,
                                    cost_fp: float = 10.0,
                                    cost_tp: float = 0.1,
                                    cost_tn: float = 0.0) -> MetricsDict:
    """
    Calculates metrics considering the cost of different types of errors.
    In the context of fraud detection, false negatives (FN) 
    tend to be more expensive than false positives (FP).
    
    Args:
        y_true: True class labels.
        y_pred: Predicted class labels
        cost_fn: Cost of missing a fraudulent transaction (false negative)
        cost_fp: Cost of false alarm (false positive)
        cost_tp: Cost of processing a true positive result
        cost_tn: Cost of processing a true negative result
        
    Returns:
        Dictionary with calculated cost metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Total cost of errors and correct predictions
    total_cost = (fn * cost_fn + fp * cost_fp + tp * cost_tp + tn * cost_tn)
    
    # Average cost per transaction
    avg_cost_per_transaction = total_cost / (tn + fp + fn + tp)
    
    # Total cost if we always predicted normal transactions
    baseline_cost = sum(y_true) * cost_fn
    
    # Savings compared to the baseline strategy
    cost_savings = baseline_cost - total_cost
    
    # Percentage of savings
    if baseline_cost > 0:
        savings_percentage = (cost_savings / baseline_cost) * 100
    else:
        savings_percentage = 0.0
    
    cost_metrics = {
        'total_cost': total_cost,
        'avg_cost_per_transaction': avg_cost_per_transaction,
        'baseline_cost': baseline_cost,
        'cost_savings': cost_savings,
        'savings_percentage': savings_percentage
    }
    
    logger.info(f"Cost-sensitive metrics calculated: "
                f"avg_cost={avg_cost_per_transaction:.2f}, "
                f"savings={savings_percentage:.2f}%")
    
    return cost_metrics


def log_metrics_to_database(engine: Engine, 
                           metrics: MetricsDict,
                           timestamp: Optional[datetime] = None) -> None:
    """
    Saves the model metrics to the database.
    
    Args:
        engine: SQLAlchemy engine to connect to the database
        metrics: Dictionary with metrics
        timestamp: Timestamp (if None, current time is used)
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now()
        
    logger.info(f"Logging metrics to database for time: {timestamp}")
    
    # Make sure we have all the metrics we need
    required_metrics = ['precision', 'recall', 'f1', 'accuracy']
    for metric in required_metrics:
        if metric not in metrics:
            logger.error(f"Required metric {metric} missing from metrics dictionary")
            return
    
    # Save the metrics to the database
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO model_metrics (time, precision, recall, f1, accuracy)
            VALUES (:time, :precision, :recall, :f1, :accuracy)
        """),
            {"time": timestamp,
             "precision": float(metrics['precision']),
             "recall": float(metrics['recall']),
             "f1": float(metrics['f1']),
             "accuracy": float(metrics['accuracy'])})
    
    logger.info("Metrics successfully saved to database")


def get_metrics_history(engine: Engine, 
                       days: int = 7) -> pd.DataFrame:
    """
    Gets the history of metrics from the database for the specified period.
    
    Args:
        engine: SQLAlchemy engine to connect to the database
        days: Number of days of history to retrieve
        
    Returns:
        DataFrame with metrics history
    """
    logger.info(f"Retrieving metrics history for the last {days} days")
    
    # Query to retrieve metrics from the database
    query = text("""
        SELECT time, precision, recall, f1, accuracy
        FROM model_metrics
        WHERE time >= NOW() - INTERVAL :days DAY
        ORDER BY time ASC
    """)
    
    # Executing a request
    with engine.connect() as connection:
        result = connection.execute(query, {"days": days})
        metrics_history = pd.DataFrame(result.fetchall())
        if not metrics_history.empty:
            metrics_history.columns = result.keys()
    
    if metrics_history.empty:
        logger.warning("No metrics found in the database for the specified period")
        return pd.DataFrame(columns=['time', 'precision', 'recall', 'f1', 'accuracy'])
    
    logger.info(f"Retrieved {len(metrics_history)} metric records")
    
    return metrics_history


def plot_metrics_history(metrics_history: pd.DataFrame,
                        metrics_to_plot: List[str] = ['precision', 'recall', 'f1', 'accuracy'],
                        figsize: Tuple[int, int] = (12, 6),
                        save_path: Optional[str] = None) -> None:
    """
    Plots the change in metrics over time.
    
    Args:
        metrics_history: DataFrame with metrics history
        metrics_to_plot: List of metrics to display
        figsize: Graph size
        save_path: Path to save the plot (if None, does not save)
    """
    if metrics_history.empty:
        logger.warning("Cannot plot empty metrics history")
        return
    
    plt.figure(figsize=figsize)
    
    for metric in metrics_to_plot:
        if metric in metrics_history.columns:
            plt.plot(metrics_history['time'], metrics_history[metric], label=metric)
    
    plt.title('Model Metrics Over Time')
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics history plot saved to {save_path}")
    
    plt.close()


def plot_roc_curve(y_true: np.ndarray,
                  y_prob: np.ndarray,
                  figsize: Tuple[int, int] = (8, 6),
                  save_path: Optional[str] = None) -> float:
    """
    Plots the ROC curve and calculates the area under the curve (AUC).
    
    Args:
        y_true: True class labels
        y_prob: Probabilities of belonging to a positive class
        figsize: Graph size
        save_path: Path to save the graph (if None, does not save)
        
    Returns:
        AUC-ROC value
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    
    plt.close()
    
    return roc_auc


def plot_precision_recall_curve(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               figsize: Tuple[int, int] = (8, 6),
                               save_path: Optional[str] = None) -> float:
    """
    Constructs an accuracy-completeness curve and calculates the average precision (AP).
    
    Args:
        y_true: True class labels
        y_prob: Probabilities of being a member of a positive class
        figsize: Graph size
        save_path: Path to save the graph (if None, does not save)
        
    Returns:
        Average Precision (AP) value
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='green', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to {save_path}")
    
    plt.close()
    
    return avg_precision


def calculate_threshold_metrics(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               thresholds: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Calculates metrics for different probability thresholds.
    
    Args:
        y_true: True class labels
        y_prob: Probabilities of belonging to a positive class
        thresholds: List of thresholds to test (if None, generates automatically)
        
    Returns:
        DataFrame with metrics for each threshold value
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics for the current threshold
        metrics = {
            'threshold': threshold,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def plot_threshold_metrics(threshold_metrics: pd.DataFrame,
                          figsize: Tuple[int, int] = (12, 6),
                          save_path: Optional[str] = None) -> None:
    """
    Plots the dependence of the metrics on the threshold value.
    
    Args:
        threshold_metrics: DataFrame with metrics for different thresholds
        figsize: The size of the graph
        save_path: Path to save the graph (if None, does not save)
    """
    plt.figure(figsize=figsize)
    
    metrics_to_plot = ['precision', 'recall', 'f1', 'accuracy']
    for metric in metrics_to_plot:
        if metric in threshold_metrics.columns:
            plt.plot(threshold_metrics['threshold'], threshold_metrics[metric], label=metric)
    
    plt.title('Metrics vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Threshold metrics plot saved to {save_path}")
    
    plt.close()


def detect_metrics_drift(metrics_history: pd.DataFrame,
                        window_size: int = 10,
                        threshold: float = 0.05) -> Dict[str, bool]:
    """
    Determines if there is drift of metrics over time.
    
    Args:
        metrics_history: DataFrame with metrics history
        window_size: Window size for moving average calculation
        threshold: Threshold value for drift detection
        
    Returns:
        Dictionary with results of drift detection by metrics
    """
    if len(metrics_history) < window_size * 2:
        logger.warning("Not enough data points for drift detection")
        return {metric: False for metric in ['precision', 'recall', 'f1', 'accuracy']}
    
    drift_results = {}
    metrics_to_check = ['precision', 'recall', 'f1', 'accuracy']
    
    for metric in metrics_to_check:
        if metric not in metrics_history.columns:
            continue
            
        # Calculating the moving average
        rolling_mean = metrics_history[metric].rolling(window=window_size, min_periods=1).mean()
        
        # Calculate the drift as the difference between the current and previous average
        current_mean = rolling_mean.iloc[-1]
        previous_mean = rolling_mean.iloc[-window_size-1] if len(rolling_mean) > window_size else rolling_mean.iloc[0]
        
        # Determining the presence of drift
        drift = abs(current_mean - previous_mean) > threshold
        drift_results[metric] = drift
        
        if drift:
            logger.warning(f"Detected drift in {metric}: current={current_mean:.4f}, previous={previous_mean:.4f}")
    
    return drift_results


def calculate_alert_score(metrics: MetricsDict, 
                         weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculates a single metric for monitoring and generating alerts.
    
    Args:
        metrics: Dictionary with metrics
        weights: Dictionary with weights for each metric (if None, uses default weights)
        
    Returns:
        The overall metric for the alerts
    """
    if weights is None:
        weights = {
            'precision': 0.4,
            'recall': 0.3,
            'f1': 0.2,
            'accuracy': 0.1
        }
    
    # Verify that all necessary metrics are in place
    for metric in weights.keys():
        if metric not in metrics:
            logger.error(f"Required metric {metric} missing for alert score calculation")
            return 0.0
    
    # Calculate the weighted sum
    alert_score = sum(metrics[metric] * weight for metric, weight in weights.items())
    
    logger.info(f"Alert score calculated: {alert_score:.4f}")
    
    return alert_score


def get_latest_metrics(engine: Engine) -> Optional[MetricsDict]:
    """
    Gets the most recently recorded metrics from the database.
    
    Args:
        Engine: SQLAlchemy engine to connect to the database
        
    Returns:
        Dictionary with the latest metrics or None if no metrics are found
    """
    logger.info("Retrieving latest metrics from database")
    
    query = text("""
        SELECT precision, recall, f1, accuracy
        FROM model_metrics
        ORDER BY time DESC
        LIMIT 1
    """)
    
    with engine.connect() as connection:
        result = connection.execute(query)
        metrics_row = result.fetchone()
    
    if metrics_row is None:
        logger.warning("No metrics found in the database")
        return None
    
    metrics = {
        'precision': float(metrics_row[0]),
        'recall': float(metrics_row[1]),
        'f1': float(metrics_row[2]),
        'accuracy': float(metrics_row[3])
    }
    
    logger.info(f"Retrieved latest metrics: {metrics}")
    
    return metrics
