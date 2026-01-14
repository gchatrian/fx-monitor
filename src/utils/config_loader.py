"""Configuration loader utility."""

import os
import yaml
from typing import Any, Dict, List


class ConfigLoader:
    """Loads and manages application configuration."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.load_config()

    def load_config(self, config_path: str = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default path relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(base_dir, "config.yaml")

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    @property
    def bloomberg_host(self) -> str:
        return self._config.get('bloomberg', {}).get('host', 'localhost')

    @property
    def bloomberg_port(self) -> int:
        return self._config.get('bloomberg', {}).get('port', 8194)

    @property
    def bloomberg_timeout(self) -> int:
        return self._config.get('bloomberg', {}).get('timeout', 30000)

    @property
    def data_directory(self) -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = self._config.get('data', {}).get('directory', './data/')
        return os.path.normpath(os.path.join(base_dir, data_dir))

    @property
    def options_file(self) -> str:
        return os.path.join(self.data_directory,
                          self._config.get('data', {}).get('options_file', 'options_portfolio.csv'))

    @property
    def forwards_file(self) -> str:
        return os.path.join(self.data_directory,
                          self._config.get('data', {}).get('forwards_file', 'forwards_blotter.csv'))

    @property
    def fx_crosses(self) -> List[str]:
        return self._config.get('fx_crosses', ['EURUSD'])

    @property
    def forward_tenors(self) -> List[str]:
        return self._config.get('forward_tenors', ['1M', '3M', '6M', '1Y'])

    @property
    def vol_tenors(self) -> List[str]:
        return self._config.get('vol_tenors', ['1M', '3M', '6M', '1Y'])

    @property
    def vol_pillars(self) -> List[str]:
        return self._config.get('vol_pillars', ['ATM', '25RR', '10RR', '25BF', '10BF'])

    @property
    def decimal_places(self) -> int:
        return self._config.get('display', {}).get('decimal_places', 4)

    @property
    def refresh_interval(self) -> int:
        return self._config.get('display', {}).get('refresh_interval', 60)

    @property
    def gamma_bump_pct(self) -> float:
        return self._config.get('risk', {}).get('gamma_bump_pct', 1.0)

    @property
    def vega_bump_bp(self) -> float:
        return self._config.get('risk', {}).get('vega_bump_bp', 100)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key path (dot notation)."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
