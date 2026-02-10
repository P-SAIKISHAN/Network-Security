import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

import os, sys
import numpy as np 
import pickle 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.ndarray):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.ndarray data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load numpy array data from file
    file_path: str location of file to load
    return: np.ndarray data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """
    Save object to file using pickle
    file_path: str location of file to save
    obj: object to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Load object from file using pickle
    file_path: str location of file to load
    return: object loaded
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate multiple models with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        models: Dictionary of models to evaluate
        params: Dictionary of hyperparameters for each model
    
    Returns:
        report: Dictionary containing model performance metrics
    """
    try:
        report = {}
        
        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")
            
            # Get parameters for current model
            param = params.get(model_name, {})
            
            if param:
                # Perform GridSearchCV if parameters are provided
                gs = GridSearchCV(model, param, cv=3, scoring='f1', n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)
                
                # Set model to best estimator
                model = gs.best_estimator_
                logging.info(f"{model_name} best parameters: {gs.best_params_}")
            else:
                # Train model without GridSearch
                model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            train_f1 = f1_score(y_train, y_train_pred, average='weighted')
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            train_precision = precision_score(y_train, y_train_pred, average='weighted')
            test_precision = precision_score(y_test, y_test_pred, average='weighted')
            
            train_recall = recall_score(y_train, y_train_pred, average='weighted')
            test_recall = recall_score(y_test, y_test_pred, average='weighted')
            
            # Store results
            report[model_name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_f1_score': train_f1,
                'test_f1_score': test_f1,
                'train_precision': train_precision,
                'test_precision': test_precision,
                'train_recall': train_recall,
                'test_recall': test_recall,
                'model': model
            }
            
            logging.info(f"{model_name} - Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
        
        return report
        
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e