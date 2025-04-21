"""
The module contains functions for data preprocessing in a fraudulent transaction detection system.
It includes loading, cleaning, scaling data and generating a synthetic transaction stream.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
from datetime import datetime

# Configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defining data types
TransactionData = pd.DataFrame
ModelData = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]  # X_train, X_test, y_train, y_test


def load_transaction_data(file_path: str) -> TransactionData:
    logger.info(f"Loading data from {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        data = pd.read_csv(file_path)
        # Lowercase column names for consistency
        data.columns = data.columns.str.lower()
        logger.info(f"Loaded {len(data)} records")
        
        # Deleting rows with missing values
        data = data.dropna()
        logger.info(f"After cleaning: {len(data)} records remain")
        
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def prepare_data_for_model(data: TransactionData, 
                          test_size: float = 0.2, 
                          random_state: int = 42) -> Tuple[ModelData, StandardScaler, List[str]]:
    """
    Prepares data for training the model: splits into training and test sets, scales the features.
    """
                            
    logger.info("Preparing data for model training")
    
    # Defining features for scaling
    columns_to_scale = [
        'v1','v2','v3','v4','v5','v6','v7','v8','v9','v10',
        'v11','v12','v13','v14','v15','v16','v17','v18','v19',
        'v20','v21','v22','v23','v24','v25','v26','v27','v28',
        'amount'
    ]
    
    # Check that all the necessary columns are present
    missing_columns = [col for col in columns_to_scale if col not in data.columns]
    if missing_columns:
        logger.warning(f"Missing columns in data: {missing_columns}")
        # Filter out missing columns
        columns_to_scale = [col for col in columns_to_scale if col in data.columns]
    
    X = data[columns_to_scale]
    y = data['class']
    
    # Division into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    logger.info(f"Data split into training ({len(X_train)} samples) and test ({len(X_test)} samples) sets")
    
    return (X_train, X_test, y_train, y_test), scaler, columns_to_scale


def generate_synthetic_data(base_data: TransactionData, 
                           num_samples: int = 1000, 
                           start_date: str = '2025-01-01',
                           frequency: str = '5min') -> TransactionData:
    """
    Generates synthetic transaction data based on the statistical characteristics of the of the underlying dataset.
    """
                             
    logger.info(f"Generating {num_samples} synthetic transactions")
    
    # Time series for transactions
    time_stamps = pd.date_range(start=start_date, periods=num_samples, freq=frequency)
    
    # Determining the proportion of fraudulent transactions
    fraud_proportion = base_data['class'].value_counts(normalize=True).get(1, 0.001)
    
    # Statistics for normal transactions
    normal_mean_V = base_data[base_data['class'] == 0].loc[:, 'v1':'v28'].mean()
    normal_std_V = base_data[base_data['class'] == 0].loc[:, 'v1':'v28'].std()
    normal_mean_amount = base_data[base_data['class'] == 0]['amount'].mean()
    normal_std_amount = base_data[base_data['class'] == 0]['amount'].std()
    
    # Statistics for fraudulent transactions
    # If there are no fraudulent transactions, we use normal statistics with a small deviation
    if (base_data['class'] == 1).sum() > 0:
        fraud_mean_V = base_data[base_data['class'] == 1].loc[:, 'v1':'v28'].mean()
        fraud_std_V = base_data[base_data['class'] == 1].loc[:, 'v1':'v28'].std()
        fraud_mean_amount = base_data[base_data['class'] == 1]['amount'].mean()
        fraud_std_amount = base_data[base_data['class'] == 1]['amount'].std()
    else:
        logger.warning("No fraud transactions in base data, using modified normal statistics")
        fraud_mean_V = normal_mean_V * 1.2
        fraud_std_V = normal_std_V * 1.5
        fraud_mean_amount = normal_mean_amount * 3
        fraud_std_amount = normal_std_amount * 2
    
    # Number of normal and fraudulent transactions
    normal_samples = int(num_samples * (1 - fraud_proportion))
    fraud_samples = num_samples - normal_samples
    
    # Generating attributes for normal transactions
    normal_transactions_V = np.random.normal(
        normal_mean_V, normal_std_V, (normal_samples, 28)
    )
    normal_amounts = np.abs(np.random.normal(
        normal_mean_amount, normal_std_amount, normal_samples
    ))
    
    # Generating indicators for fraudulent transactions
    fraud_transactions_V = np.random.normal(
        fraud_mean_V, fraud_std_V, (fraud_samples, 28)
    )
    fraud_amounts = np.abs(np.random.normal(
        fraud_mean_amount, fraud_std_amount, fraud_samples
    ))
    
    # Data merging
    synthetic_V = np.vstack([normal_transactions_V, fraud_transactions_V])
    synthetic_amount = np.hstack([normal_amounts, fraud_amounts])
    synthetic_class = np.hstack([np.zeros(normal_samples), np.ones(fraud_samples)])
    
    # DataFrame creation
    synthetic_data = pd.DataFrame(synthetic_V, columns=[f'v{i}' for i in range(1, 29)])
    synthetic_data['amount'] = synthetic_amount
    synthetic_data['class'] = synthetic_class.astype(int)
    synthetic_data.index = time_stamps
    
    # Shuffle the data and reset the index
    synthetic_data = synthetic_data.sample(frac=1).reset_index()
    synthetic_data.rename(columns={'index': 'time'}, inplace=True)
    
    logger.info(f"Generated {normal_samples} normal and {fraud_samples} fraud transactions")
    
    return synthetic_data


def simulate_transaction_stream(data_source: TransactionData, 
                               num_samples: int = 10, 
                               fraud_probability: float = 0.02) -> TransactionData:
    """
    Simulates transaction flow to test the system in real time.
    """
                                 
    logger.info(f"Simulating transaction stream with {num_samples} samples")
    
    new_data_list = []
    for _ in range(num_samples):
        # Generation of 28 features with little random noise
        new_transaction = pd.DataFrame(
            np.random.normal(size=(1, 28)),
            columns=[f'v{i}' for i in range(1, 29)]
        )
        
        # Generation of transaction amount
        new_transaction['amount'] = np.abs(np.random.normal(size=1) * 100 + 50)
        
        # Defining a transaction class with a given fraud probability
        new_transaction['class'] = np.random.choice(
            [0, 1], 
            size=1, 
            p=[1-fraud_probability, fraud_probability]
        )
        
        new_data_list.append(new_transaction)
    
    # Merge all transactions
    stream_data = pd.concat(new_data_list, ignore_index=True)
    stream_data['time'] = pd.Timestamp.now()
    
    return stream_data


def log_statistics_to_database(engine: Engine, data: TransactionData, timestamp: Optional[datetime] = None) -> None:
    """
    Saves the statistical characteristics of the data to the database.
    
    Args:
        engine: SQLAlchemy engine to connect to the database
        data: DataFrame with data for calculating statistics
        timestamp: Time stamp for statistics (if None, current time is used)
    """
  
    if timestamp is None:
        timestamp = pd.Timestamp.now()
        
    logger.info(f"Logging statistics to database for time: {timestamp}")
    
    # Select only numeric columns for statistics
    numeric_cols = data.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        # Skip the class columns
        if col != 'class' and col != 'predicted_class':
            mean_val = data[col].mean()
            std_val = data[col].std()
            
            # Save to database
            with engine.begin() as connection:
                connection.execute(text("""
                    INSERT INTO data_statistics (time, feature, mean, std)
                    VALUES (:time, :feature, :mean, :std)
                """),
                    {"time": timestamp, "feature": col, "mean": float(mean_val), "std": float(std_val)})
    
    logger.info(f"Statistics for {len(numeric_cols)} features saved to database")


def detect_data_drift(engine: Engine, 
                     current_data: TransactionData, 
                     reference_period: str = '1 day',
                     p_threshold: float = 0.05) -> Dict[str, bool]:
    """
    Determines if there is data drift by comparing current data to historical data.
    
    Args:
        Engine: SQLAlchemy engine to connect to the database
        current_data: DataFrame with current data
        reference_period: Period for sampling historical data
        p_threshold: Threshold p-value for Kolmogorov-Smirnov test
        
    Returns:
        Dictionary with the results of feature drift detection
    """
                       
    logger.info("Checking for data drift")
    
    drift_results = {}
    
    # Retrieving historical data from the database
    with engine.connect() as connection:
        reference_data = pd.read_sql(text(f"""
            SELECT * FROM transactions 
            WHERE time >= NOW() - INTERVAL '{reference_period}'
        """), connection)
    
    if len(reference_data) < 10:
        logger.warning("Not enough historical data for drift detection")
        return {"error": "Not enough historical data"}
    
    # We check the drift for each feature
    numeric_cols = current_data.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        if col not in ['class', 'predicted_class'] and col in reference_data.columns:
            # Running a statistical test
            stat, p_value = kstest(
                current_data[col].values,
                reference_data[col].values
            )
            
            # Determining the presence of drift
            drift_detected = p_value < p_threshold
            drift_results[col] = drift_detected
            
            if drift_detected:
                logger.warning(f"Data drift detected in feature {col} (p-value: {p_value:.4f})")
    
    return drift_results
