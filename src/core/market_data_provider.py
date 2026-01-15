"""Market data provider with Bloomberg and fallback mock data."""

from datetime import datetime, date
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging
import random

from ..utils.config_loader import ConfigLoader
from ..utils.date_utils import DateUtils

logger = logging.getLogger(__name__)


@dataclass
class FXMarketData:
    """Container for FX market data."""
    cross: str
    spot: float = 0.0
    forward_points: Dict[str, float] = field(default_factory=dict)
    forward_rates: Dict[str, float] = field(default_factory=dict)
    vol_surface: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MarketDataProvider:
    """
    Market data provider with Bloomberg API and mock data fallback.

    Automatically falls back to realistic mock data when Bloomberg is unavailable.
    """

    # Realistic base market data (as of early 2025)
    MOCK_SPOT_RATES = {
        'EURUSD': 1.0850,
        'GBPUSD': 1.2720,
        'USDJPY': 148.50,
        'EURGBP': 0.8530,
        'EURCHF': 0.9420,
        'USDCAD': 1.3580,
        'AUDUSD': 0.6620,
        'AUDNZD': 1.1050,
        'USDCHF': 0.8680,
        'NZDUSD': 0.5990,
    }

    # Forward points in pips (annualized interest rate differential effect)
    MOCK_FORWARD_POINTS_BASE = {
        'EURUSD': {'1W': -2, '2W': -4, '1M': -8, '2M': -16, '3M': -24, '6M': -48, '9M': -72, '1Y': -95},
        'GBPUSD': {'1W': -1, '2W': -3, '1M': -6, '2M': -12, '3M': -18, '6M': -36, '9M': -54, '1Y': -70},
        'USDJPY': {'1W': 3, '2W': 6, '1M': 12, '2M': 24, '3M': 38, '6M': 78, '9M': 115, '1Y': 155},
        'EURGBP': {'1W': -0.5, '2W': -1, '1M': -2, '2M': -4, '3M': -6, '6M': -12, '9M': -18, '1Y': -24},
        'EURCHF': {'1W': 1, '2W': 2, '1M': 4, '2M': 8, '3M': 12, '6M': 24, '9M': 36, '1Y': 48},
        'USDCAD': {'1W': 0.5, '2W': 1, '1M': 2, '2M': 4, '3M': 6, '6M': 12, '9M': 18, '1Y': 24},
        'AUDUSD': {'1W': -1, '2W': -2, '1M': -4, '2M': -8, '3M': -12, '6M': -24, '9M': -36, '1Y': -48},
        'AUDNZD': {'1W': 0.2, '2W': 0.4, '1M': 1, '2M': 2, '3M': 3, '6M': 6, '9M': 9, '1Y': 12},
    }

    # Base ATM volatilities by tenor (in %)
    MOCK_ATM_VOL = {
        'EURUSD': {'1W': 7.5, '2W': 7.8, '1M': 8.0, '2M': 8.2, '3M': 8.3, '6M': 8.5, '9M': 8.6, '1Y': 8.7},
        'GBPUSD': {'1W': 8.5, '2W': 8.8, '1M': 9.0, '2M': 9.2, '3M': 9.4, '6M': 9.6, '9M': 9.7, '1Y': 9.8},
        'USDJPY': {'1W': 9.0, '2W': 9.3, '1M': 9.5, '2M': 9.8, '3M': 10.0, '6M': 10.3, '9M': 10.5, '1Y': 10.7},
        'EURGBP': {'1W': 6.0, '2W': 6.2, '1M': 6.5, '2M': 6.7, '3M': 6.8, '6M': 7.0, '9M': 7.1, '1Y': 7.2},
        'EURCHF': {'1W': 5.5, '2W': 5.7, '1M': 6.0, '2M': 6.2, '3M': 6.3, '6M': 6.5, '9M': 6.6, '1Y': 6.7},
        'USDCAD': {'1W': 6.5, '2W': 6.7, '1M': 7.0, '2M': 7.2, '3M': 7.3, '6M': 7.5, '9M': 7.6, '1Y': 7.7},
        'AUDUSD': {'1W': 10.0, '2W': 10.3, '1M': 10.5, '2M': 10.8, '3M': 11.0, '6M': 11.3, '9M': 11.5, '1Y': 11.7},
        'AUDNZD': {'1W': 8.0, '2W': 8.2, '1M': 8.5, '2M': 8.7, '3M': 8.8, '6M': 9.0, '9M': 9.1, '1Y': 9.2},
    }

    # Typical RR and BF values (skew and smile)
    MOCK_VOL_SKEW = {
        'EURUSD': {'25RR': -0.3, '10RR': -0.6, '25BF': 0.15, '10BF': 0.45},
        'GBPUSD': {'25RR': -0.4, '10RR': -0.8, '25BF': 0.20, '10BF': 0.55},
        'USDJPY': {'25RR': 0.5, '10RR': 1.0, '25BF': 0.25, '10BF': 0.65},
        'EURGBP': {'25RR': -0.2, '10RR': -0.4, '25BF': 0.10, '10BF': 0.30},
        'EURCHF': {'25RR': 0.3, '10RR': 0.6, '25BF': 0.20, '10BF': 0.50},
        'USDCAD': {'25RR': -0.2, '10RR': -0.4, '25BF': 0.12, '10BF': 0.35},
        'AUDUSD': {'25RR': -0.5, '10RR': -1.0, '25BF': 0.25, '10BF': 0.65},
        'AUDNZD': {'25RR': -0.1, '10RR': -0.2, '25BF': 0.08, '10BF': 0.25},
    }

    def __init__(self):
        self.config = ConfigLoader()
        self._bloomberg_client = None
        self._use_bloomberg = False
        self._market_data_cache: Dict[str, FXMarketData] = {}
        self._last_refresh: Optional[datetime] = None

        # Try to initialize Bloomberg
        self._try_init_bloomberg()

    def _try_init_bloomberg(self):
        """Try to initialize Bloomberg connection."""
        try:
            import blpapi
            from .bloomberg_client import BloombergClient
            self._bloomberg_client = BloombergClient()
            if self._bloomberg_client.connect():
                self._use_bloomberg = True
                logger.info("Bloomberg API connected successfully")
            else:
                logger.info("Bloomberg not available, using mock data")
                self._use_bloomberg = False
        except ImportError:
            logger.info("blpapi not installed, using mock data")
            self._use_bloomberg = False
        except Exception as e:
            logger.warning(f"Bloomberg initialization failed: {e}, using mock data")
            self._use_bloomberg = False

    def is_using_bloomberg(self) -> bool:
        """Check if using real Bloomberg data."""
        return self._use_bloomberg

    def connect(self) -> bool:
        """Connect to data source."""
        if self._use_bloomberg and self._bloomberg_client:
            return self._bloomberg_client.connect()
        return True  # Mock is always "connected"

    def disconnect(self) -> None:
        """Disconnect from data source."""
        if self._bloomberg_client:
            self._bloomberg_client.disconnect()

    def is_connected(self) -> bool:
        """Check connection status."""
        if self._use_bloomberg and self._bloomberg_client:
            return self._bloomberg_client.is_connected()
        return True  # Mock is always connected

    def get_spot_rate(self, cross: str) -> float:
        """Get spot rate for a cross."""
        if self._use_bloomberg and self._bloomberg_client:
            rate = self._bloomberg_client.get_spot_rate(cross)
            if rate:
                return rate

        # Mock data with small random variation
        base_rate = self.MOCK_SPOT_RATES.get(cross.upper(), 1.0)
        variation = base_rate * 0.001 * (random.random() - 0.5)
        return base_rate + variation

    def get_forward_points(self, cross: str) -> Dict[str, float]:
        """Get forward points for a cross."""
        if self._use_bloomberg and self._bloomberg_client:
            points = self._bloomberg_client.get_forward_points(cross)
            if points:
                return points

        # Mock forward points
        base_points = self.MOCK_FORWARD_POINTS_BASE.get(cross.upper(), {})
        result = {}
        for tenor, pts in base_points.items():
            variation = pts * 0.05 * (random.random() - 0.5)
            result[tenor] = pts + variation
        return result

    def get_volatility_surface(self, cross: str) -> Dict[str, Dict[str, float]]:
        """Get volatility surface for a cross."""
        if self._use_bloomberg and self._bloomberg_client:
            surface = self._bloomberg_client.get_volatility_surface(cross)
            if surface:
                return surface

        # Build mock vol surface
        cross_upper = cross.upper()
        atm_vols = self.MOCK_ATM_VOL.get(cross_upper, {})
        skew = self.MOCK_VOL_SKEW.get(cross_upper, {'25RR': 0, '10RR': 0, '25BF': 0.15, '10BF': 0.45})

        vol_surface = {}
        for tenor, atm in atm_vols.items():
            # Add small random variation
            atm_var = atm * 0.02 * (random.random() - 0.5)
            vol_surface[tenor] = {
                'ATM': atm + atm_var,
                '25RR': skew['25RR'] + 0.05 * (random.random() - 0.5),
                '10RR': skew['10RR'] + 0.1 * (random.random() - 0.5),
                '25BF': skew['25BF'] + 0.02 * (random.random() - 0.5),
                '10BF': skew['10BF'] + 0.05 * (random.random() - 0.5),
            }

        return vol_surface

    def get_all_market_data(self, cross: str, force_refresh: bool = False) -> FXMarketData:
        """
        Get all market data for a cross.

        Args:
            cross: FX pair (e.g., 'EURUSD')
            force_refresh: Force data refresh

        Returns:
            FXMarketData object
        """
        cross = cross.upper()

        if not force_refresh and cross in self._market_data_cache:
            cached = self._market_data_cache[cross]
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age < self.config.refresh_interval:
                return cached

        market_data = FXMarketData(cross=cross)

        # Get spot
        market_data.spot = self.get_spot_rate(cross)

        # Get forward points
        market_data.forward_points = self.get_forward_points(cross)

        # Calculate forward rates
        pip_factor = 10000 if 'JPY' not in cross else 100
        for tenor, points in market_data.forward_points.items():
            market_data.forward_rates[tenor] = market_data.spot + (points / pip_factor)

        # Get vol surface
        market_data.vol_surface = self.get_volatility_surface(cross)

        market_data.timestamp = datetime.now()
        self._market_data_cache[cross] = market_data
        self._last_refresh = datetime.now()

        return market_data

    def refresh_all_crosses(self, crosses: List[str] = None) -> Dict[str, FXMarketData]:
        """Refresh market data for all crosses."""
        if crosses is None:
            crosses = self.config.fx_crosses

        result = {}
        for cross in crosses:
            result[cross] = self.get_all_market_data(cross, force_refresh=True)

        return result

    def get_forward_rate_for_date(self, cross: str, value_date: date) -> float:
        """Get interpolated forward rate for a specific value date."""
        market_data = self.get_all_market_data(cross)

        if not market_data.spot or not market_data.forward_points:
            return market_data.spot

        today = date.today()
        days_to_value = (value_date - today).days

        if days_to_value <= 0:
            return market_data.spot

        # Convert tenors to days and sort
        tenor_days = []
        for tenor, points in market_data.forward_points.items():
            try:
                years = DateUtils.tenor_to_years(tenor)
                days = int(years * 365)
                tenor_days.append((days, points))
            except:
                continue

        tenor_days.sort(key=lambda x: x[0])

        if not tenor_days:
            return market_data.spot

        pip_factor = 10000 if 'JPY' not in cross else 100

        # Interpolation logic
        if days_to_value <= tenor_days[0][0]:
            ratio = days_to_value / tenor_days[0][0]
            interp_points = tenor_days[0][1] * ratio
            return market_data.spot + (interp_points / pip_factor)

        if days_to_value >= tenor_days[-1][0]:
            if len(tenor_days) >= 2:
                d1, p1 = tenor_days[-2]
                d2, p2 = tenor_days[-1]
                slope = (p2 - p1) / (d2 - d1)
                interp_points = p2 + slope * (days_to_value - d2)
            else:
                interp_points = tenor_days[-1][1]
            return market_data.spot + (interp_points / pip_factor)

        for i in range(len(tenor_days) - 1):
            d1, p1 = tenor_days[i]
            d2, p2 = tenor_days[i + 1]

            if d1 <= days_to_value <= d2:
                ratio = (days_to_value - d1) / (d2 - d1)
                interp_points = p1 + ratio * (p2 - p1)
                return market_data.spot + (interp_points / pip_factor)

        return market_data.spot


# Singleton instance
_market_data_provider = None


def get_market_data_provider() -> MarketDataProvider:
    """Get the singleton market data provider instance."""
    global _market_data_provider
    if _market_data_provider is None:
        _market_data_provider = MarketDataProvider()
    return _market_data_provider
