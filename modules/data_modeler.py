import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import time

# XGBoost import
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Darts import for time series forecasting
try:
    from darts import TimeSeries
    from darts.models import (
        Prophet, ExponentialSmoothing, ARIMA,
        LinearRegressionModel, NaiveDrift
    )
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False

# GARCH import
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

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
                                          f"CV RMSE: {cv_rmse:.4f} (�{cv_scores.std():.4f})")

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
                                          f"CV Accuracy: {cv_accuracy:.4f} (�{cv_scores.std():.4f})")

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


class TraditionalMLEnsemble:
    """Traditional ML Ensemble using RandomForest, XGBoost, AdaBoost, and Linear Regression."""

    def __init__(self, logger):
        self.logger = logger
        self.models = {}
        self.ensemble_model = None
        self.is_trained = False

    def create_models(self, **kwargs):
        """Create individual models for the ensemble."""
        models = []

        # RandomForest
        rf_params = kwargs.get('rf_params', {'n_estimators': 100, 'random_state': 42})
        models.append(('rf', RandomForestRegressor(**rf_params)))

        # XGBoost (if available)
        if XGB_AVAILABLE:
            xgb_params = kwargs.get('xgb_params', {'n_estimators': 100, 'random_state': 42})
            models.append(('xgb', xgb.XGBRegressor(**xgb_params)))
        else:
            # Fallback to GradientBoosting
            gb_params = kwargs.get('gb_params', {'n_estimators': 100, 'random_state': 42})
            models.append(('gb', GradientBoostingRegressor(**gb_params)))

        # AdaBoost
        ada_params = kwargs.get('ada_params', {'n_estimators': 100, 'random_state': 42})
        models.append(('ada', AdaBoostRegressor(**ada_params)))

        # Linear Regression
        models.append(('lr', LinearRegression()))

        self.models = dict(models)
        return models

    def train(self, X_train, y_train, **kwargs):
        """Train the ensemble model."""
        self.logger.info("Training Traditional ML Ensemble")

        # Create models
        model_list = self.create_models(**kwargs)

        # Create voting regressor
        self.ensemble_model = VotingRegressor(estimators=model_list)

        # Train the ensemble
        self.ensemble_model.fit(X_train, y_train)
        self.is_trained = True

        self.logger.info("Traditional ML Ensemble training completed")
        return self.ensemble_model

    def predict(self, X_test):
        """Make predictions using the ensemble."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.ensemble_model.predict(X_test)

    def get_individual_predictions(self, X_test):
        """Get predictions from individual models."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        predictions = {}
        for name, model in self.ensemble_model.named_estimators_.items():
            predictions[name] = model.predict(X_test)
        return predictions


class DeepLearningEnsemble:
    """Deep Learning Ensemble using LSTM, 1D CNN, TCN, and Dense Network."""

    def __init__(self, logger, sequence_length=60):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for DeepLearningEnsemble")

        self.logger = logger
        self.sequence_length = sequence_length
        self.models = {}
        self.ensemble_model = None
        self.is_trained = False

    def create_lstm_model(self, input_shape, **kwargs):
        """Create LSTM model."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def create_cnn_model(self, input_shape, **kwargs):
        """Create 1D CNN model."""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def create_tcn_model(self, input_shape, **kwargs):
        """Create Temporal Convolutional Network."""
        # Simplified TCN using dilated convolutions
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, dilation_rate=1, activation='relu', input_shape=input_shape),
            Conv1D(filters=32, kernel_size=3, dilation_rate=2, activation='relu'),
            Conv1D(filters=32, kernel_size=3, dilation_rate=4, activation='relu'),
            Conv1D(filters=32, kernel_size=3, dilation_rate=8, activation='relu'),
            layers.GlobalMaxPooling1D(),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def create_dense_model(self, input_shape, **kwargs):
        """Create Dense Network."""
        # Flatten the input for dense layers
        model = Sequential([
            layers.Flatten(input_shape=input_shape),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def prepare_sequences(self, data):
        """Prepare data sequences for time series models."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def train(self, X_train, y_train, epochs=50, batch_size=32, **kwargs):
        """Train the deep learning ensemble."""
        self.logger.info("Training Deep Learning Ensemble")

        # Prepare sequence data
        if len(X_train.shape) == 2:
            # Reshape for sequence modeling
            X_train_seq = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        else:
            X_train_seq = X_train

        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

        # Create models
        self.models['lstm'] = self.create_lstm_model(input_shape, **kwargs)
        self.models['cnn'] = self.create_cnn_model(input_shape, **kwargs)
        self.models['tcn'] = self.create_tcn_model(input_shape, **kwargs)
        self.models['dense'] = self.create_dense_model(input_shape, **kwargs)

        # Train individual models
        self.trained_models = {}
        for name, model in self.models.items():
            self.logger.info(f"Training {name} model")
            history = model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size,
                              validation_split=0.2, verbose=0)
            self.trained_models[name] = model

        self.is_trained = True
        self.logger.info("Deep Learning Ensemble training completed")
        return self.trained_models

    def predict(self, X_test):
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare sequence data
        if len(X_test.shape) == 2:
            X_test_seq = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        else:
            X_test_seq = X_test

        # Get predictions from all models
        predictions = []
        for name, model in self.trained_models.items():
            pred = model.predict(X_test_seq, verbose=0)
            predictions.append(pred.flatten())

        # Average the predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred

    def get_individual_predictions(self, X_test):
        """Get predictions from individual models."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if len(X_test.shape) == 2:
            X_test_seq = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        else:
            X_test_seq = X_test

        predictions = {}
        for name, model in self.trained_models.items():
            pred = model.predict(X_test_seq, verbose=0)
            predictions[name] = pred.flatten()
        return predictions


class HybridMLDLEnsemble:
    """Hybrid ML/DL Ensemble combining RandomForest, LSTM, and XGBoost."""

    def __init__(self, logger, sequence_length=60):
        self.logger = logger
        self.sequence_length = sequence_length
        self.rf_model = None
        self.lstm_model = None
        self.meta_model = None
        self.is_trained = False

    def create_lstm_model(self, input_shape):
        """Create LSTM model for temporal features."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM")

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train, y_train, epochs=50, **kwargs):
        """Train the hybrid ensemble."""
        self.logger.info("Training Hybrid ML/DL Ensemble")

        # Train RandomForest on original features
        rf_params = kwargs.get('rf_params', {'n_estimators': 100, 'random_state': 42})
        self.rf_model = RandomForestRegressor(**rf_params)
        self.rf_model.fit(X_train, y_train)

        # Prepare sequence data for LSTM
        if len(X_train.shape) == 2:
            X_train_seq = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        else:
            X_train_seq = X_train

        # Train LSTM
        if TF_AVAILABLE:
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            self.lstm_model = self.create_lstm_model(input_shape)
            self.lstm_model.fit(X_train_seq, y_train, epochs=epochs, verbose=0)

        # Get predictions from both models for meta-learning
        rf_pred = self.rf_model.predict(X_train)
        if TF_AVAILABLE and self.lstm_model:
            lstm_pred = self.lstm_model.predict(X_train_seq, verbose=0).flatten()
        else:
            lstm_pred = np.zeros_like(rf_pred)

        # Create meta-features
        meta_features = np.column_stack([rf_pred, lstm_pred])

        # Train meta-model (XGBoost if available, otherwise GradientBoosting)
        if XGB_AVAILABLE:
            self.meta_model = xgb.XGBRegressor(**kwargs.get('xgb_params', {'n_estimators': 100, 'random_state': 42}))
        else:
            self.meta_model = GradientBoostingRegressor(**kwargs.get('gb_params', {'n_estimators': 100, 'random_state': 42}))

        self.meta_model.fit(meta_features, y_train)
        self.is_trained = True

        self.logger.info("Hybrid ML/DL Ensemble training completed")
        return self

    def predict(self, X_test):
        """Make predictions using the hybrid ensemble."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Get RF predictions
        rf_pred = self.rf_model.predict(X_test)

        # Get LSTM predictions
        if TF_AVAILABLE and self.lstm_model:
            if len(X_test.shape) == 2:
                X_test_seq = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            else:
                X_test_seq = X_test
            lstm_pred = self.lstm_model.predict(X_test_seq, verbose=0).flatten()
        else:
            lstm_pred = np.zeros_like(rf_pred)

        # Create meta-features
        meta_features = np.column_stack([rf_pred, lstm_pred])

        # Final prediction from meta-model
        final_pred = self.meta_model.predict(meta_features)
        return final_pred


class SpecializedFinancialEnsemble:
    """Specialized Financial Ensemble using Darts time series models, LSTM, RandomForest, and GARCH."""

    def __init__(self, logger, sequence_length=60):
        self.logger = logger
        self.sequence_length = sequence_length
        self.darts_models = {}
        self.lstm_model = None
        self.rf_model = None
        self.garch_model = None
        self.is_trained = False
        self.time_series = None

    def prepare_darts_data(self, dates, prices):
        """Prepare data for Darts time series models."""
        try:
            if not DARTS_AVAILABLE:
                self.logger.warning("Darts not available")
                return None

            if dates is None or prices is None:
                self.logger.warning("Dates or prices are None")
                return None

            if len(dates) != len(prices):
                self.logger.warning("Dates and prices have different lengths")
                return None

            if len(dates) < 10:  # Minimum data points for forecasting
                self.logger.warning("Insufficient data points for time series forecasting")
                return None

            # Convert to pandas DataFrame first
            df = pd.DataFrame({
                'date': pd.to_datetime(dates),
                'value': prices
            })

            # Remove duplicates and sort by date
            df = df.drop_duplicates(subset=['date']).sort_values('date')
            df = df.set_index('date')

            # Remove any rows with NaN values
            df = df.dropna()

            if len(df) < 10:
                self.logger.warning("Insufficient valid data points after cleaning")
                return None

            # Create Darts TimeSeries object
            ts = TimeSeries.from_dataframe(df, value_cols=['value'])

            self.logger.info(f"Created Darts TimeSeries with {len(ts)} data points")
            return ts

        except Exception as e:
            self.logger.warning(f"Failed to create Darts TimeSeries: {e}")
            return None

    def create_darts_models(self):
        """Create and configure Darts forecasting models."""
        models = {}

        if not DARTS_AVAILABLE:
            self.logger.warning("Darts not available, skipping time series models")
            return models

        try:
            # Prophet model (via Darts wrapper)
            models['prophet'] = Prophet()

            # Exponential Smoothing for trend analysis
            models['exp_smoothing'] = ExponentialSmoothing()

            # ARIMA for statistical forecasting
            models['arima'] = ARIMA()

            # Naive drift as baseline
            models['naive_drift'] = NaiveDrift()

            # Linear regression for multivariate capability (if more features added later)
            models['linear_reg'] = LinearRegressionModel(lags=5)

        except Exception as e:
            self.logger.warning(f"Failed to create some Darts models: {e}")

        return models

    def train(self, X_train, y_train, dates=None, epochs=50, **kwargs):
        """Train the specialized financial ensemble."""
        self.logger.info("Training Specialized Financial Ensemble")

        # Train RandomForest for multi-feature predictions
        rf_params = kwargs.get('rf_params', {'n_estimators': 100, 'random_state': 42})
        self.rf_model = RandomForestRegressor(**rf_params)
        self.rf_model.fit(X_train, y_train)

        # Train LSTM for temporal patterns
        if TF_AVAILABLE:
            if len(X_train.shape) == 2:
                X_train_seq = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            else:
                X_train_seq = X_train

            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            self.lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            self.lstm_model.compile(optimizer='adam', loss='mse')
            self.lstm_model.fit(X_train_seq, y_train, epochs=epochs, verbose=0)

        # Train Darts time series models (if available and dates provided)
        if DARTS_AVAILABLE and dates is not None:
            try:
                # Prepare time series data
                self.time_series = self.prepare_darts_data(dates, y_train)

                if self.time_series is not None:
                    # Create and train multiple Darts models
                    darts_model_configs = self.create_darts_models()

                    for model_name, model in darts_model_configs.items():
                        try:
                            self.logger.info(f"Training Darts {model_name} model")
                            model.fit(self.time_series)
                            self.darts_models[model_name] = model
                            self.logger.info(f"Successfully trained {model_name}")
                        except Exception as e:
                            self.logger.warning(f"Failed to train {model_name}: {e}")
                            continue

                    if self.darts_models:
                        self.logger.info(f"Successfully trained {len(self.darts_models)} Darts models")
                    else:
                        self.logger.warning("No Darts models were successfully trained")

            except Exception as e:
                self.logger.warning(f"Darts training failed: {e}")
                self.darts_models = {}

        # Train GARCH for volatility modeling (if available)
        if ARCH_AVAILABLE:
            try:
                # Use price returns for GARCH
                returns = np.diff(y_train) / y_train[:-1] * 100  # Percentage returns
                self.garch_model = arch_model(returns, vol='GARCH', p=1, q=1)
                self.garch_fitted = self.garch_model.fit(disp='off')
            except Exception as e:
                self.logger.warning(f"GARCH training failed: {e}")
                self.garch_model = None

        self.is_trained = True
        self.logger.info("Specialized Financial Ensemble training completed")
        return self

    def predict(self, X_test, future_dates=None, periods=1):
        """Make predictions using the specialized financial ensemble."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        predictions = []
        weights = []

        # RandomForest prediction
        rf_pred = self.rf_model.predict(X_test)
        predictions.append(rf_pred)
        weights.append(0.4)

        # LSTM prediction
        if TF_AVAILABLE and self.lstm_model:
            if len(X_test.shape) == 2:
                X_test_seq = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            else:
                X_test_seq = X_test
            lstm_pred = self.lstm_model.predict(X_test_seq, verbose=0).flatten()
            predictions.append(lstm_pred)
            weights.append(0.4)

        # Darts time series predictions (trend/seasonality component)
        if self.darts_models and periods > 0:
            try:
                darts_predictions = []
                darts_weights = []

                for model_name, model in self.darts_models.items():
                    try:
                        # Make forecast for the specified number of periods
                        forecast = model.predict(n=periods)
                        forecast_values = forecast.values().flatten()

                        # Ensure prediction length matches other predictions
                        if len(forecast_values) == len(rf_pred):
                            darts_predictions.append(forecast_values)
                            # Assign weights based on model type
                            if model_name == 'prophet':
                                darts_weights.append(0.3)  # Higher weight for Prophet
                            elif model_name == 'exp_smoothing':
                                darts_weights.append(0.25)
                            elif model_name == 'arima':
                                darts_weights.append(0.25)
                            else:
                                darts_weights.append(0.1)  # Lower weight for others

                        self.logger.info(f"Successfully generated predictions from {model_name}")

                    except Exception as e:
                        self.logger.warning(f"Prediction failed for {model_name}: {e}")
                        continue

                # If we have Darts predictions, combine them
                if darts_predictions:
                    # Normalize weights
                    darts_weights = np.array(darts_weights)
                    darts_weights = darts_weights / np.sum(darts_weights)

                    # Weighted average of Darts predictions
                    combined_darts_pred = np.average(darts_predictions, axis=0, weights=darts_weights)
                    predictions.append(combined_darts_pred)
                    weights.append(0.3)  # Weight for combined Darts prediction

                    self.logger.info(f"Combined predictions from {len(darts_predictions)} Darts models")

            except Exception as e:
                self.logger.warning(f"Darts predictions failed: {e}")

        # Normalize weights
        weights = np.array(weights) / np.sum(weights)

        # Weighted average of predictions
        if len(predictions) > 1:
            final_pred = np.average(predictions, axis=0, weights=weights)
        else:
            final_pred = predictions[0]

        return final_pred

    def predict_volatility(self, horizon=1):
        """Predict volatility using GARCH model."""
        if self.garch_model and hasattr(self, 'garch_fitted'):
            try:
                forecast = self.garch_fitted.forecast(horizon=horizon)
                return forecast.variance.values[-1, :]
            except Exception as e:
                self.logger.warning(f"GARCH volatility prediction failed: {e}")
                return None
        return None


class EnsembleManager:
    """Manager class for all ensemble approaches."""

    def __init__(self, logger):
        self.logger = logger
        self.ensembles = {}

    def create_ensemble(self, ensemble_type, **kwargs):
        """Create and return an ensemble of the specified type."""
        if ensemble_type == 'traditional_ml':
            ensemble = TraditionalMLEnsemble(self.logger)
        elif ensemble_type == 'deep_learning':
            ensemble = DeepLearningEnsemble(self.logger, **kwargs)
        elif ensemble_type == 'hybrid':
            ensemble = HybridMLDLEnsemble(self.logger, **kwargs)
        elif ensemble_type == 'financial':
            ensemble = SpecializedFinancialEnsemble(self.logger, **kwargs)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")

        self.ensembles[ensemble_type] = ensemble
        return ensemble

    def get_ensemble(self, ensemble_type):
        """Get an existing ensemble."""
        return self.ensembles.get(ensemble_type)

    def compare_ensembles(self, X_test, y_test):
        """Compare performance of all trained ensembles."""
        results = {}

        for name, ensemble in self.ensembles.items():
            if hasattr(ensemble, 'is_trained') and ensemble.is_trained:
                try:
                    predictions = ensemble.predict(X_test)

                    # Calculate metrics
                    mse = mean_squared_error(y_test, predictions)
                    mae = mean_absolute_error(y_test, predictions)
                    rmse = np.sqrt(mse)

                    results[name] = {
                        'mse': mse,
                        'mae': mae,
                        'rmse': rmse,
                        'predictions': predictions
                    }

                    self.logger.info(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

                except Exception as e:
                    self.logger.error(f"Error evaluating {name}: {e}")
                    results[name] = {'error': str(e)}

        return results

    def get_trained_darts_models(self) -> Dict[str, str]:
        """Get information about successfully trained Darts models."""
        model_info = {}
        for model_name, model in self.darts_models.items():
            model_info[model_name] = str(type(model).__name__)
        return model_info

    def is_darts_available(self) -> bool:
        """Check if Darts is available and models are trained."""
        return DARTS_AVAILABLE and len(self.darts_models) > 0

    def validate_darts_integration(self) -> Dict[str, Any]:
        """Validate the Darts integration and return status information."""
        validation_results = {
            'darts_library_available': DARTS_AVAILABLE,
            'time_series_created': self.time_series is not None,
            'models_trained': len(self.darts_models),
            'trained_model_names': list(self.darts_models.keys()),
            'integration_status': 'success' if self.is_darts_available() else 'failed'
        }

        if DARTS_AVAILABLE:
            try:
                # Test basic Darts functionality
                test_series = TimeSeries.from_values(np.random.randn(30))
                test_model = NaiveDrift()
                test_model.fit(test_series)
                test_pred = test_model.predict(n=5)
                validation_results['basic_functionality_test'] = 'passed'
            except Exception as e:
                validation_results['basic_functionality_test'] = f'failed: {e}'
        else:
            validation_results['basic_functionality_test'] = 'skipped - Darts not available'

        return validation_results


class ModelingError(Exception):
    """Exception raised for modeling-related errors."""
    pass