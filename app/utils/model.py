"""
The module contains functions for working with a machine learning model 
in the fraudulent transaction detection system.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, Union, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_score, 
    recall_score, f1_score, confusion_matrix, accuracy_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
from datetime import datetime

# Configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definition of types
TransactionData = pd.DataFrame
Model = RandomForestClassifier
ModelMetrics = Dict[str, float]


def create_fraud_detection_model(class_weight: str = 'balanced', 
                                n_estimators: int = 100,
                                random_state: int = 42) -> Model:
    """
    Creates a random forest model for fraud detection.
    
    Args:
        class_weight: Class weighting strategy (‘balanced’ for unbalanced data)
        n_estimators: Number of trees in the ensemble
        random_state: Value to initialise the random number generator
        
    Returns:
        Initialised but not trained RandomForestClassifier model
    """
                                  
    logger.info(f"Creating RandomForest model with {n_estimators} trees")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,  # Использовать все доступные ядра процессора
        verbose=0
    )
    
    return model


def train_model(model: Model, 
               X_train: Union[np.ndarray, pd.DataFrame], 
               y_train: Union[np.ndarray, pd.Series]) -> Model:
    """
    Trains a model on the provided data.
    
    Args:
        model: RandomForestClassifier model to be trained
        X_train: Training features
        y_train: Training class labels
        
    Returns:
        Trained model
    """
                 
    logger.info(f"Training model on {len(X_train)} samples")
    
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    return model


def update_model(model: Model, 
                new_X: Union[np.ndarray, pd.DataFrame], 
                new_y: Union[np.ndarray, pd.Series]) -> Model:
    """
    Retrains an existing model on new data.
    
    Args:
        model: Previously trained RandomForestClassifier model
        new_X: New class labels for pre-training
        new_y: New class labels for pre-training
        
    Returns:
        Pre-trained model
    """
    logger.info(f"Updating model with {len(new_X)} new samples")
    
    # Retraining the model on new data
    model.fit(new_X, new_y)
    
    logger.info("Model update completed")
    
    return model


def evaluate_model(model: Model, 
                  X_test: Union[np.ndarray, pd.DataFrame], 
                  y_test: Union[np.ndarray, pd.Series]) -> ModelMetrics:
    """
    Evaluates the quality of the model on test data.
    
    Args:
        model: Trained RandomForestClassifier model
        X_test: Test features
        y_test: Test class labels
        
    Returns:
        Dictionary with model quality metrics
    """
    logger.info(f"Evaluating model on {len(X_test)} test samples")
    
    # Getting predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculation of metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    logger.info(f"Model metrics: accuracy={metrics['accuracy']:.4f}, "
                f"precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"f1={metrics['f1']:.4f}, "
                f"roc_auc={metrics['roc_auc']:.4f}")
    
    return metrics


def print_classification_report(y_true: Union[np.ndarray, pd.Series], 
                               y_pred: np.ndarray) -> None:
    """
    Outputs a classification report.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
    """
    report = classification_report(y_true, y_pred)
    logger.info(f"\nClassification Report:\n{report}")


def plot_confusion_matrix(y_true: Union[np.ndarray, pd.Series], 
                         y_pred: np.ndarray, 
                         save_path: Optional[str] = None) -> None:
    """
    Constructs an error matrix and saves it to a file.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        save_path: Path to save the image (if None, does not save)
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_feature_importance(model: Model, 
                           feature_names: List[str], 
                           top_n: int = 10,
                           save_path: Optional[str] = None) -> None:
    """
  Constructs a graph of feature importance for the model.
    
    Args:
        model: The trained RandomForestClassifier model
        feature_names: List of feature names
        top_n: Number of most important features to display
        save_path: Path to save the image (if None, does not save)
    """
    # Obtaining the importance of attributes
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Top {top_n} Feature Importances')
    plt.bar(range(top_n), importances[indices], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.close()


def log_metrics_to_database(engine: Engine, 
                           metrics: ModelMetrics,
                           timestamp: Optional[datetime] = None) -> None:
    """
    Saves the model metrics to the database.
    
    Args:
        engine: SQLAlchemy engine to connect to the database
        metrics: Dictionary with model metrics
        timestamp: Timestamp (if None, current time is used)
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now()
        
    logger.info(f"Logging metrics to database: {metrics}")
    
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


def predict_transaction(model: Model, 
                       scaler: StandardScaler,
                       transaction_data: TransactionData) -> np.ndarray:
    """
    Makes predictions for new transactions.
    
    Args:
        model: Trained RandomForestClassifier model
        scaler: Trained StandardScaler for scaling features
        Transaction_data: DataFrame with transaction data for classification
        
    Returns:
        Array with predicted classes for each transaction
    """
                         
    logger.info(f"Predicting classes for {len(transaction_data)} transactions")
    
    # Get the features, excluding class labels and predictions, if any
    features = transaction_data.drop(['class', 'predicted_class'], axis=1, errors='ignore')
    if 'time' in features.columns:
        features = features.drop('time', axis=1)
    
    # Scaling the attributes
    scaled_features = scaler.transform(features)
    
    # Making predictions
    predictions = model.predict(scaled_features)
    
    logger.info(f"Predicted {sum(predictions)} fraud transactions out of {len(predictions)}")
    
    return predictions


def get_prediction_probabilities(model: Model, 
                                scaler: StandardScaler,
                                transaction_data: TransactionData) -> np.ndarray:
    """
    Gets the probabilities of fraud class membership for transactions.
    
    Args:
        model: Trained RandomForestClassifier model
        scaler: Trained StandardScaler to scale the features
        transaction_data: DataFrame with transaction data
        
    Returns:
        Array with probabilities of belonging to a fraud class
    """
    # Get the features, excluding class labels and predictions, if any
    features = transaction_data.drop(['class', 'predicted_class'], axis=1, errors='ignore')
    if 'time' in features.columns:
        features = features.drop('time', axis=1)
    
    # Scaling the attributes
    scaled_features = scaler.transform(features)
    
    # We get the probabilities
    probabilities = model.predict_proba(scaled_features)[:, 1]
    
    return probabilities


def save_model(model: Model, 
              scaler: StandardScaler, 
              model_path: str = 'models/fraud_model.pkl',
              scaler_path: str = 'models/scaler.pkl') -> None:
    """
    Saves the model and scaler to files.
    
    Args:
        model: Trained RandomForestClassifier model to save
        scaler: Trained StandardScaler to save.
        model_path: Path to save the model
        scaler_path: Path to save the scaler
    """
    # Create a directory for models, if there is no directory for models
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    
    # Saving the scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")


def load_model(model_path: str = 'models/fraud_model.pkl',
              scaler_path: str = 'models/scaler.pkl') -> Tuple[Model, StandardScaler]:
    """
    Loads the model and scaler from the files.
    
    Args:
        model_path: Path to the saved model
        scaler_path: Path to the saved scaler.
        
    Returns:
        A tuple of the loaded model and scaler
        
    Raises:
        FileNotFoundError: If no model or scaler files are found
    """
    # Checking the availability of files
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    # Loading the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    
    # Loading the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"Scaler loaded from {scaler_path}")
    
    return model, scaler
