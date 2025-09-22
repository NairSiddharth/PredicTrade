import json
import os
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Manages configuration settings for the stock predictor application."""

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                raise FileNotFoundError(f"Configuration file {self.config_path} not found")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to configuration value (e.g., "data_sources.google_trends.enabled")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def save(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")

    def get_data_sources_config(self) -> Dict[str, Any]:
        """Get data sources configuration."""
        return self.get("data_sources", {})

    def get_ml_config(self) -> Dict[str, Any]:
        """Get machine learning configuration."""
        return self.get("machine_learning", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get("logging", {})

    def get_api_settings(self) -> Dict[str, Any]:
        """Get API settings configuration."""
        return self.get("api_settings", {})


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass