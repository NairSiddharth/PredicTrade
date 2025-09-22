import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from .logger import StockPredictorLogger
from .config_manager import ConfigManager


class ModelEvaluator:
    """Handles model evaluation, metrics calculation, and performance visualization."""

    def __init__(self, config_manager: ConfigManager, logger: StockPredictorLogger):
        """
        Initialize the model evaluator.

        Args:
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        self.config = config_manager
        self.logger = logger.get_logger("data_evaluation")
        self.evaluation_results = {}

    def evaluate_regression_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str = "model") -> Dict[str, float]:
        """
        Evaluate regression model performance.

        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model being evaluated

        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.log_model_operation("Regression Evaluation", f"Evaluating {model_name}")

        try:
            # Calculate regression metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            # Calculate additional metrics
            mape = self._calculate_mape(y_true, y_pred)
            directional_accuracy = self._calculate_directional_accuracy(y_true, y_pred)

            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'mape': mape,
                'directional_accuracy': directional_accuracy,
                'n_samples': len(y_true)
            }

            # Store results
            self.evaluation_results[f"{model_name}_regression"] = metrics

            self.logger.log_model_operation("Regression Evaluation Success",
                                          f"RMSE: {rmse:.4f}, R�: {r2:.4f}, MAPE: {mape:.2f}%")

            return metrics

        except Exception as e:
            self.logger.log_error(e, f"evaluate_regression_model - {model_name}")
            return {}

    def evaluate_classification_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    y_pred_proba: Optional[np.ndarray] = None,
                                    model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate classification model performance.

        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            y_pred_proba: Predicted class probabilities (optional)
            model_name: Name of the model being evaluated

        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.log_model_operation("Classification Evaluation", f"Evaluating {model_name}")

        try:
            # Calculate classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm.tolist(),
                'n_samples': len(y_true)
            }

            # Add ROC AUC for binary classification
            if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
                try:
                    # Use probability of positive class
                    if y_pred_proba.ndim > 1:
                        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        roc_auc = roc_auc_score(y_true, y_pred_proba)
                    metrics['roc_auc'] = roc_auc
                except Exception as e:
                    self.logger.warning(f"Could not calculate ROC AUC: {str(e)}")

            # Store results
            self.evaluation_results[f"{model_name}_classification"] = metrics

            self.logger.log_model_operation("Classification Evaluation Success",
                                          f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}")

            return metrics

        except Exception as e:
            self.logger.log_error(e, f"evaluate_classification_model - {model_name}")
            return {}

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            MAPE as percentage
        """
        try:
            # Avoid division by zero
            mask = y_true != 0
            if not np.any(mask):
                return float('inf')

            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            return mape

        except Exception:
            return float('inf')

    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (percentage of correct direction predictions).

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Directional accuracy as percentage
        """
        try:
            if len(y_true) < 2:
                return 0.0

            # Calculate changes
            true_changes = np.diff(y_true)
            pred_changes = np.diff(y_pred)

            # Count correct directions
            correct_directions = np.sum(np.sign(true_changes) == np.sign(pred_changes))
            total_predictions = len(true_changes)

            return (correct_directions / total_predictions) * 100

        except Exception:
            return 0.0

    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare performance of multiple models.

        Args:
            model_results: Dictionary with model names and their metrics

        Returns:
            DataFrame comparing model performance
        """
        self.logger.log_model_operation("Model Comparison", f"Comparing {len(model_results)} models")

        try:
            comparison_data = []

            for model_name, metrics in model_results.items():
                row = {'model': model_name}
                row.update(metrics)
                comparison_data.append(row)

            comparison_df = pd.DataFrame(comparison_data)

            # Sort by best performing metric
            if 'rmse' in comparison_df.columns:
                comparison_df = comparison_df.sort_values('rmse')
            elif 'accuracy' in comparison_df.columns:
                comparison_df = comparison_df.sort_values('accuracy', ascending=False)

            self.logger.log_model_operation("Model Comparison Success",
                                          f"Generated comparison for {len(comparison_df)} models")

            return comparison_df

        except Exception as e:
            self.logger.log_error(e, "compare_models")
            return pd.DataFrame()

    def calculate_feature_correlation(self, X: np.ndarray, feature_names: List[str],
                                    target: np.ndarray) -> pd.DataFrame:
        """
        Calculate correlation matrix between features and target.

        Args:
            X: Feature matrix
            feature_names: List of feature names
            target: Target vector

        Returns:
            Correlation DataFrame
        """
        self.logger.log_data_operation("Correlation Analysis", f"Analyzing {len(feature_names)} features")

        try:
            # Create DataFrame with features and target
            df = pd.DataFrame(X, columns=feature_names)
            df['target'] = target

            # Calculate correlation matrix
            correlation_matrix = df.corr()

            # Extract correlations with target
            target_correlations = correlation_matrix['target'].drop('target')
            target_correlations = target_correlations.sort_values(key=abs, ascending=False)

            self.logger.log_data_operation("Correlation Analysis Success",
                                         f"Computed correlations for {len(feature_names)} features")

            return target_correlations.to_frame('correlation')

        except Exception as e:
            self.logger.log_error(e, "calculate_feature_correlation")
            return pd.DataFrame()

    def analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = "model") -> Dict[str, Any]:
        """
        Analyze residuals for regression models.

        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model

        Returns:
            Dictionary with residual analysis results
        """
        self.logger.log_model_operation("Residual Analysis", f"Analyzing residuals for {model_name}")

        try:
            residuals = y_true - y_pred

            analysis = {
                'mean_residual': np.mean(residuals),
                'std_residual': np.std(residuals),
                'min_residual': np.min(residuals),
                'max_residual': np.max(residuals),
                'q25_residual': np.percentile(residuals, 25),
                'q75_residual': np.percentile(residuals, 75)
            }

            # Test for normality (simple check)
            analysis['residuals_skewness'] = self._calculate_skewness(residuals)

            self.logger.log_model_operation("Residual Analysis Success",
                                          f"Mean residual: {analysis['mean_residual']:.4f}")

            return analysis

        except Exception as e:
            self.logger.log_error(e, f"analyze_residuals - {model_name}")
            return {}

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            skewness = np.mean(((data - mean) / std) ** 3)
            return skewness
        except Exception:
            return 0.0

    def generate_evaluation_report(self, model_name: str) -> str:
        """
        Generate a comprehensive evaluation report for a model.

        Args:
            model_name: Name of the model to report on

        Returns:
            Formatted report string
        """
        try:
            if model_name not in self.evaluation_results:
                return f"No evaluation results found for {model_name}"

            metrics = self.evaluation_results[model_name]
            report_lines = [f"Evaluation Report for {model_name}", "=" * 50]

            if 'rmse' in metrics:  # Regression model
                report_lines.extend([
                    f"Root Mean Square Error: {metrics['rmse']:.4f}",
                    f"Mean Absolute Error: {metrics['mae']:.4f}",
                    f"R� Score: {metrics['r2_score']:.4f}",
                    f"Mean Absolute Percentage Error: {metrics['mape']:.2f}%",
                    f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%"
                ])
            elif 'accuracy' in metrics:  # Classification model
                report_lines.extend([
                    f"Accuracy: {metrics['accuracy']:.4f}",
                    f"Precision: {metrics['precision']:.4f}",
                    f"Recall: {metrics['recall']:.4f}",
                    f"F1 Score: {metrics['f1_score']:.4f}"
                ])
                if 'roc_auc' in metrics:
                    report_lines.append(f"ROC AUC: {metrics['roc_auc']:.4f}")

            report_lines.append(f"Number of samples: {metrics['n_samples']}")

            return "\n".join(report_lines)

        except Exception as e:
            self.logger.log_error(e, f"generate_evaluation_report - {model_name}")
            return f"Error generating report for {model_name}"

    def export_results(self, filepath: str, format: str = 'json') -> bool:
        """
        Export evaluation results to file.

        Args:
            filepath: Path to save results
            format: Format to save ('json' or 'csv')

        Returns:
            True if successful, False otherwise
        """
        try:
            if format == 'json':
                import json
                # Convert numpy arrays to lists for JSON serialization
                results_serializable = {}
                for key, value in self.evaluation_results.items():
                    results_serializable[key] = {}
                    for metric_key, metric_value in value.items():
                        if isinstance(metric_value, np.ndarray):
                            results_serializable[key][metric_key] = metric_value.tolist()
                        else:
                            results_serializable[key][metric_key] = metric_value

                with open(filepath, 'w') as f:
                    json.dump(results_serializable, f, indent=2)

            elif format == 'csv':
                # Convert to DataFrame and save as CSV
                comparison_df = self.compare_models(self.evaluation_results)
                comparison_df.to_csv(filepath, index=False)

            self.logger.log_model_operation("Export Results", f"Exported to {filepath} in {format} format")
            return True

        except Exception as e:
            self.logger.log_error(e, f"export_results to {filepath}")
            return False

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluation results."""
        return {
            'models_evaluated': list(self.evaluation_results.keys()),
            'total_models': len(self.evaluation_results),
            'results': self.evaluation_results.copy()
        }


class EvaluationError(Exception):
    """Exception raised for evaluation-related errors."""
    pass