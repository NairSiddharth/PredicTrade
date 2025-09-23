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
            smape = self._calculate_smape(y_true, y_pred)
            mase = self._calculate_mase(y_true, y_pred)
            directional_accuracy = self._calculate_directional_accuracy(y_true, y_pred)

            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'mape': mape,
                'smape': smape,
                'mase': mase,
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

    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            sMAPE as percentage
        """
        try:
            # Calculate sMAPE: 100 * mean(|actual - forecast| / ((|actual| + |forecast|) / 2))
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

            # Avoid division by zero
            mask = denominator != 0
            if not np.any(mask):
                return float('inf')

            smape = 100 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])
            return smape

        except Exception:
            return float('inf')

    def _calculate_mase(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Scaled Error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            MASE value
        """
        try:
            if len(y_true) < 2:
                return float('inf')

            # Calculate MAE of predictions
            mae_forecast = np.mean(np.abs(y_true - y_pred))

            # Calculate MAE of naive forecast (using previous period)
            naive_forecast = y_true[:-1]
            actual_values = y_true[1:]
            mae_naive = np.mean(np.abs(actual_values - naive_forecast))

            # Avoid division by zero
            if mae_naive == 0:
                return 0.0 if mae_forecast == 0 else float('inf')

            mase = mae_forecast / mae_naive
            return mase

        except Exception:
            return float('inf')

    def evaluate_financial_trading_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                             dates: Optional[np.ndarray] = None,
                                             model_name: str = "model",
                                             risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Evaluate model performance from a financial trading perspective.

        Args:
            y_true: True price values
            y_pred: Predicted price values
            dates: Optional array of dates for time-based calculations
            model_name: Name of the model being evaluated
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Dictionary containing financial trading metrics
        """
        self.logger.log_model_operation("Financial Trading Evaluation", f"Evaluating {model_name}")

        try:
            if len(y_true) < 2:
                return {}

            # Calculate price returns
            actual_returns = np.diff(y_true) / y_true[:-1]
            predicted_returns = np.diff(y_pred) / y_pred[:-1]

            # Hit rate (directional accuracy) - already calculated in directional_accuracy
            hit_rate = self._calculate_directional_accuracy(y_true, y_pred) / 100

            # Calculate strategy returns (if you followed model predictions)
            strategy_returns = actual_returns * np.sign(predicted_returns)

            # Sharpe Ratio
            daily_risk_free_rate = risk_free_rate / 252  # Convert annual to daily
            excess_returns = strategy_returns - daily_risk_free_rate
            sharpe_ratio = np.mean(excess_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) != 0 else 0

            # Sortino Ratio (downside deviation)
            downside_returns = strategy_returns[strategy_returns < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
            sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation != 0 else 0

            # Maximum Drawdown
            cumulative_returns = np.cumprod(1 + strategy_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)

            # Win Rate (percentage of profitable trades)
            win_rate = np.mean(strategy_returns > 0)

            # Average Win/Loss
            winning_trades = strategy_returns[strategy_returns > 0]
            losing_trades = strategy_returns[strategy_returns < 0]
            avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
            avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

            # Profit Factor
            gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0
            gross_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

            # Total Return and Volatility
            total_return = np.prod(1 + strategy_returns) - 1
            annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            annualized_volatility = np.std(strategy_returns) * np.sqrt(252)

            # Calmar Ratio (return/max drawdown)
            calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0

            # Information Ratio (excess return over benchmark per unit of tracking error)
            # Using buy-and-hold as benchmark
            benchmark_returns = actual_returns
            excess_vs_benchmark = strategy_returns - benchmark_returns
            tracking_error = np.std(excess_vs_benchmark)
            information_ratio = np.mean(excess_vs_benchmark) / tracking_error if tracking_error != 0 else 0

            financial_metrics = {
                'hit_rate': hit_rate,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': win_loss_ratio,
                'profit_factor': profit_factor,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'n_trades': len(strategy_returns)
            }

            # Store results
            self.evaluation_results[f"{model_name}_financial"] = financial_metrics

            self.logger.log_model_operation("Financial Trading Evaluation Success",
                                          f"Hit Rate: {hit_rate:.3f}, Sharpe: {sharpe_ratio:.3f}, Max DD: {max_drawdown:.3f}")

            return financial_metrics

        except Exception as e:
            self.logger.log_error(e, f"evaluate_financial_trading_performance - {model_name}")
            return {}

    def evaluate_comprehensive_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                         dates: Optional[np.ndarray] = None,
                                         model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate model using both traditional ML metrics and financial trading metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            dates: Optional dates array
            model_name: Name of the model

        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        self.logger.log_model_operation("Comprehensive Evaluation", f"Full evaluation for {model_name}")

        try:
            # Get traditional ML metrics
            ml_metrics = self.evaluate_regression_model(y_true, y_pred, model_name)

            # Get financial trading metrics
            financial_metrics = self.evaluate_financial_trading_performance(y_true, y_pred, dates, model_name)

            # Combine results
            comprehensive_metrics = {
                'ml_metrics': ml_metrics,
                'financial_metrics': financial_metrics,
                'model_name': model_name,
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            }

            # Store combined results
            self.evaluation_results[f"{model_name}_comprehensive"] = comprehensive_metrics

            self.logger.log_model_operation("Comprehensive Evaluation Success",
                                          f"Complete evaluation for {model_name}")

            return comprehensive_metrics

        except Exception as e:
            self.logger.log_error(e, f"evaluate_comprehensive_performance - {model_name}")
            return {}

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
                    f"R² Score: {metrics['r2_score']:.4f}",
                    f"Mean Absolute Percentage Error: {metrics['mape']:.2f}%",
                    f"Symmetric MAPE: {metrics['smape']:.2f}%",
                    f"Mean Absolute Scaled Error: {metrics['mase']:.4f}",
                    f"Directional Accuracy (Hit Rate): {metrics['directional_accuracy']:.2f}%"
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