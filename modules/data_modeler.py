import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import time
from .logger import StockPredictorLogger
from .config_manager import ConfigManager


class StockPredictor:
    """Handles machine learning model training and prediction for stock data."""

    def __init__(self, config_manager: ConfigManager, logger: StockPredictorLogger):
        """
        Initialize the stock predictor.

        Args:
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        self.config = config_manager
        self.logger = logger.get_logger("data_modeler")
        self.models = {}
        self.model_performance = {}

    def train_regression_model(self, X_train: np.ndarray, y_train: np.ndarray,
                             model_type: str = 'random_forest',
                             model_params: Optional[Dict] = None) -> Any:
        """
        Train a regression model for stock price prediction.

        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of model to train
            model_params: Model parameters

        Returns:
            Trained model
        """
        self.logger.log_model_operation("Regression Training", f"Training {model_type} model")

        start_time = time.time()

        try:
            # Get model configuration
            ml_config = self.config.get_ml_config()
            default_params = ml_config.get('models', {}).get(model_type, {})

            # Merge with provided parameters
            if model_params:
                default_params.update(model_params)

            # Initialize model based on type
            if model_type == 'random_forest':
                model = RandomForestRegressor(**default_params)
            elif model_type == 'linear_regression':
                model = LinearRegression(**default_params)
            elif model_type == 'svr':
                model = SVR(**default_params)
            else:
                self.logger.warning(f"Unknown model type: {model_type}, using RandomForest")
                model = RandomForestRegressor(**default_params)

            # Train the model
            model.fit(X_train, y_train)

            # Store the trained model
            self.models[f"{model_type}_regression"] = model

            # Calculate training time
            training_time = time.time() - start_time
            self.logger.log_performance("Regression Training", training_time,
                                       f"Model: {model_type}, Samples: {len(X_train)}")

            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())

            self.model_performance[f"{model_type}_regression"] = {
                'cv_rmse': cv_rmse,
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'n_samples': len(X_train),
                'n_features': X_train.shape[1]
            }

            self.logger.log_model_operation("Regression Training Success",
                                          f"CV RMSE: {cv_rmse:.4f} (±{cv_scores.std():.4f})")

            return model

        except Exception as e:
            self.logger.log_error(e, f"train_regression_model - {model_type}")
            return None

    def train_classification_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                 model_type: str = 'random_forest',
                                 model_params: Optional[Dict] = None) -> Any:
        """
        Train a classification model for stock categorization.

        Args:
            X_train: Training features
            y_train: Training targets (categorical)
            model_type: Type of model to train
            model_params: Model parameters

        Returns:
            Trained model
        """
        self.logger.log_model_operation("Classification Training", f"Training {model_type} model")

        start_time = time.time()

        try:
            # Get model configuration
            ml_config = self.config.get_ml_config()
            default_params = ml_config.get('models', {}).get(model_type, {})

            # Merge with provided parameters
            if model_params:
                default_params.update(model_params)

            # Initialize model based on type
            if model_type == 'random_forest':
                model = RandomForestClassifier(**default_params)
            elif model_type == 'logistic_regression':
                model = LogisticRegression(**default_params)
            elif model_type == 'svc':
                model = SVC(**default_params)
            else:
                self.logger.warning(f"Unknown model type: {model_type}, using RandomForest")
                model = RandomForestClassifier(**default_params)

            # Train the model
            model.fit(X_train, y_train)

            # Store the trained model
            self.models[f"{model_type}_classification"] = model

            # Calculate training time
            training_time = time.time() - start_time
            self.logger.log_performance("Classification Training", training_time,
                                       f"Model: {model_type}, Samples: {len(X_train)}")

            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_accuracy = cv_scores.mean()

            self.model_performance[f"{model_type}_classification"] = {
                'cv_accuracy': cv_accuracy,
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'n_samples': len(X_train),
                'n_features': X_train.shape[1]
            }

            self.logger.log_model_operation("Classification Training Success",
                                          f"CV Accuracy: {cv_accuracy:.4f} (±{cv_scores.std():.4f})")

            return model

        except Exception as e:
            self.logger.log_error(e, f"train_classification_model - {model_type}")
            return None

    def predict(self, model_key: str, X_test: np.ndarray) -> Optional[np.ndarray]:
        """
        Make predictions using a trained model.

        Args:
            model_key: Key to identify the model
            X_test: Test features

        Returns:
            Predictions array
        """
        self.logger.log_model_operation("Prediction", f"Making predictions with {model_key}")

        try:
            if model_key not in self.models:
                self.logger.error(f"Model {model_key} not found")
                return None

            model = self.models[model_key]
            predictions = model.predict(X_test)

            self.logger.log_model_operation("Prediction Success",
                                          f"Generated {len(predictions)} predictions")

            return predictions

        except Exception as e:
            self.logger.log_error(e, f"predict - {model_key}")
            return None

    def predict_proba(self, model_key: str, X_test: np.ndarray) -> Optional[np.ndarray]:
        """
        Get prediction probabilities for classification models.

        Args:
            model_key: Key to identify the model
            X_test: Test features

        Returns:
            Prediction probabilities array
        """
        try:
            if model_key not in self.models:
                self.logger.error(f"Model {model_key} not found")
                return None

            model = self.models[model_key]

            if not hasattr(model, 'predict_proba'):
                self.logger.error(f"Model {model_key} does not support probability predictions")
                return None

            probabilities = model.predict_proba(X_test)

            self.logger.log_model_operation("Probability Prediction Success",
                                          f"Generated probabilities for {len(probabilities)} samples")

            return probabilities

        except Exception as e:
            self.logger.log_error(e, f"predict_proba - {model_key}")
            return None

    def create_undervalued_labels(self, stock_data: pd.DataFrame,
                                criteria: Optional[Dict] = None) -> np.ndarray:
        """
        Create binary labels for undervalued/overvalued classification.

        Args:
            stock_data: DataFrame with financial metrics
            criteria: Dictionary defining undervalued criteria

        Returns:
            Binary labels array (1 for undervalued, 0 for not undervalued)
        """
        self.logger.log_model_operation("Label Creation", "Creating undervalued classification labels")

        try:
            # Default criteria for undervalued stocks
            default_criteria = {
                'pe_ratio_threshold': 15.0,
                'pb_ratio_threshold': 1.5,
                'price_to_fcf_threshold': 15.0
            }

            if criteria:
                default_criteria.update(criteria)

            labels = np.zeros(len(stock_data))

            # Apply criteria
            conditions = []

            if 'pe_ratio' in stock_data.columns:
                pe_condition = stock_data['pe_ratio'] < default_criteria['pe_ratio_threshold']
                conditions.append(pe_condition)

            if 'pb_ratio' in stock_data.columns:
                pb_condition = stock_data['pb_ratio'] < default_criteria['pb_ratio_threshold']
                conditions.append(pb_condition)

            if 'price_to_fcf' in stock_data.columns:
                fcf_condition = stock_data['price_to_fcf'] < default_criteria['price_to_fcf_threshold']
                conditions.append(fcf_condition)

            # Stock is undervalued if it meets at least 2 out of 3 criteria
            if len(conditions) >= 2:
                combined_condition = sum(conditions) >= 2
                labels = combined_condition.astype(int)

            undervalued_count = np.sum(labels)
            total_count = len(labels)

            self.logger.log_model_operation("Label Creation Success",
                                          f"Undervalued: {undervalued_count}/{total_count} "
                                          f"({undervalued_count/total_count*100:.1f}%)")

            return labels

        except Exception as e:
            self.logger.log_error(e, "create_undervalued_labels")
            return np.zeros(len(stock_data))

    def get_feature_importance(self, model_key: str) -> Optional[np.ndarray]:
        """
        Get feature importance from tree-based models.

        Args:
            model_key: Key to identify the model

        Returns:
            Feature importance array
        """
        try:
            if model_key not in self.models:
                self.logger.error(f"Model {model_key} not found")
                return None

            model = self.models[model_key]

            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            else:
                self.logger.warning(f"Model {model_key} does not have feature_importances_")
                return None

        except Exception as e:
            self.logger.log_error(e, f"get_feature_importance - {model_key}")
            return None

    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                            model_type: str, param_grid: Dict,
                            task_type: str = 'regression') -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of model
            param_grid: Parameter grid for search
            task_type: 'regression' or 'classification'

        Returns:
            Best parameters dictionary
        """
        self.logger.log_model_operation("Hyperparameter Tuning",
                                       f"Tuning {model_type} for {task_type}")

        start_time = time.time()

        try:
            # Initialize base model
            if task_type == 'regression':
                if model_type == 'random_forest':
                    base_model = RandomForestRegressor()
                elif model_type == 'svr':
                    base_model = SVR()
                else:
                    self.logger.error(f"Unsupported regression model type: {model_type}")
                    return {}
                scoring = 'neg_mean_squared_error'
            else:  # classification
                if model_type == 'random_forest':
                    base_model = RandomForestClassifier()
                elif model_type == 'svc':
                    base_model = SVC()
                else:
                    self.logger.error(f"Unsupported classification model type: {model_type}")
                    return {}
                scoring = 'accuracy'

            # Perform grid search
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring=scoring, n_jobs=-1
            )

            grid_search.fit(X_train, y_train)

            tuning_time = time.time() - start_time
            self.logger.log_performance("Hyperparameter Tuning", tuning_time,
                                       f"Model: {model_type}, Combinations: {len(param_grid)}")

            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            self.logger.log_model_operation("Hyperparameter Tuning Success",
                                          f"Best score: {best_score:.4f}, Best params: {best_params}")

            return best_params

        except Exception as e:
            self.logger.log_error(e, f"hyperparameter_tuning - {model_type}")
            return {}

    def save_model(self, model_key: str, filepath: str) -> bool:
        """
        Save a trained model to disk.

        Args:
            model_key: Key to identify the model
            filepath: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        try:
            if model_key not in self.models:
                self.logger.error(f"Model {model_key} not found")
                return False

            model = self.models[model_key]
            joblib.dump(model, filepath)

            self.logger.log_model_operation("Model Save", f"Saved {model_key} to {filepath}")
            return True

        except Exception as e:
            self.logger.log_error(e, f"save_model - {model_key}")
            return False

    def load_model(self, model_key: str, filepath: str) -> bool:
        """
        Load a trained model from disk.

        Args:
            model_key: Key to store the model
            filepath: Path to load the model from

        Returns:
            True if successful, False otherwise
        """
        try:
            model = joblib.load(filepath)
            self.models[model_key] = model

            self.logger.log_model_operation("Model Load", f"Loaded {model_key} from {filepath}")
            return True

        except Exception as e:
            self.logger.log_error(e, f"load_model - {model_key}")
            return False

    def get_model_performance(self) -> Dict:
        """Get performance metrics for all trained models."""
        return self.model_performance.copy()


class ModelingError(Exception):
    """Exception raised for modeling-related errors."""
    pass