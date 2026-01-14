"""Bloomberg API client for FX market data."""

import blpapi
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, field

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


class BloombergClient:
    """Client for Bloomberg Desktop API to fetch FX market data."""

    def __init__(self):
        self.config = ConfigLoader()
        self.session = None
        self.ref_data_service = None
        self._connected = False
        self._market_data_cache: Dict[str, FXMarketData] = {}

    def connect(self) -> bool:
        """Establish connection to Bloomberg API."""
        if self._connected:
            return True

        try:
            session_options = blpapi.SessionOptions()
            session_options.setServerHost(self.config.bloomberg_host)
            session_options.setServerPort(self.config.bloomberg_port)

            self.session = blpapi.Session(session_options)

            if not self.session.start():
                logger.error("Failed to start Bloomberg session")
                return False

            if not self.session.openService("//blp/refdata"):
                logger.error("Failed to open //blp/refdata service")
                return False

            self.ref_data_service = self.session.getService("//blp/refdata")
            self._connected = True
            logger.info("Connected to Bloomberg API")
            return True

        except Exception as e:
            logger.error(f"Bloomberg connection error: {e}")
            return False

    def disconnect(self) -> None:
        """Close Bloomberg session."""
        if self.session:
            self.session.stop()
            self._connected = False
            logger.info("Disconnected from Bloomberg API")

    def is_connected(self) -> bool:
        """Check if connected to Bloomberg."""
        return self._connected

    def _send_request(self, request) -> Optional[List[blpapi.Message]]:
        """Send a request and collect responses."""
        if not self._connected:
            if not self.connect():
                return None

        try:
            self.session.sendRequest(request)
            messages = []

            while True:
                event = self.session.nextEvent(self.config.bloomberg_timeout)
                for msg in event:
                    messages.append(msg)

                if event.eventType() == blpapi.Event.RESPONSE:
                    break

            return messages

        except Exception as e:
            logger.error(f"Bloomberg request error: {e}")
            return None

    def get_spot_rate(self, cross: str) -> Optional[float]:
        """Fetch spot rate for an FX cross."""
        ticker = f"{cross} Curncy"

        request = self.ref_data_service.createRequest("ReferenceDataRequest")
        request.append("securities", ticker)
        request.append("fields", "PX_LAST")

        messages = self._send_request(request)
        if not messages:
            return None

        try:
            for msg in messages:
                security_data = msg.getElement("securityData")
                for i in range(security_data.numValues()):
                    sec = security_data.getValueAsElement(i)
                    field_data = sec.getElement("fieldData")
                    if field_data.hasElement("PX_LAST"):
                        return field_data.getElementAsFloat("PX_LAST")
        except Exception as e:
            logger.error(f"Error parsing spot rate for {cross}: {e}")

        return None

    def get_forward_points(self, cross: str) -> Dict[str, float]:
        """Fetch forward points curve for an FX cross."""
        forward_points = {}
        tenors = self.config.forward_tenors

        # Build tickers for each tenor
        tickers = []
        for tenor in tenors:
            # Bloomberg convention: EURUSD1M Curncy for 1M forward points
            ticker = f"{cross}{tenor} Curncy"
            tickers.append((tenor, ticker))

        request = self.ref_data_service.createRequest("ReferenceDataRequest")
        for _, ticker in tickers:
            request.append("securities", ticker)
        request.append("fields", "PX_LAST")

        messages = self._send_request(request)
        if not messages:
            return forward_points

        try:
            for msg in messages:
                security_data = msg.getElement("securityData")
                for i in range(security_data.numValues()):
                    sec = security_data.getValueAsElement(i)
                    security = sec.getElementAsString("security")

                    # Extract tenor from ticker
                    for tenor, ticker in tickers:
                        if ticker == security:
                            field_data = sec.getElement("fieldData")
                            if field_data.hasElement("PX_LAST"):
                                points = field_data.getElementAsFloat("PX_LAST")
                                forward_points[tenor] = points
                            break

        except Exception as e:
            logger.error(f"Error parsing forward points for {cross}: {e}")

        return forward_points

    def get_volatility_surface(self, cross: str) -> Dict[str, Dict[str, float]]:
        """
        Fetch volatility surface for an FX cross.

        Returns dict: {tenor: {pillar: vol}}
        Where pillar is one of: ATM, 25RR, 10RR, 25BF, 10BF
        """
        vol_surface = {}
        tenors = self.config.vol_tenors
        pillars = self.config.vol_pillars

        # Build tickers
        # Bloomberg conventions:
        # ATM: EURUSDV1M Curncy (ATM vol)
        # 25D RR: EURUSD25R1M Curncy
        # 10D RR: EURUSD10R1M Curncy
        # 25D BF: EURUSD25B1M Curncy
        # 10D BF: EURUSD10B1M Curncy

        tickers = []
        for tenor in tenors:
            for pillar in pillars:
                if pillar == "ATM":
                    ticker = f"{cross}V{tenor} Curncy"
                elif pillar == "25RR":
                    ticker = f"{cross}25R{tenor} Curncy"
                elif pillar == "10RR":
                    ticker = f"{cross}10R{tenor} Curncy"
                elif pillar == "25BF":
                    ticker = f"{cross}25B{tenor} Curncy"
                elif pillar == "10BF":
                    ticker = f"{cross}10B{tenor} Curncy"
                else:
                    continue

                tickers.append((tenor, pillar, ticker))

        request = self.ref_data_service.createRequest("ReferenceDataRequest")
        for _, _, ticker in tickers:
            request.append("securities", ticker)
        request.append("fields", "PX_LAST")

        messages = self._send_request(request)
        if not messages:
            return vol_surface

        try:
            for msg in messages:
                security_data = msg.getElement("securityData")
                for i in range(security_data.numValues()):
                    sec = security_data.getValueAsElement(i)
                    security = sec.getElementAsString("security")

                    for tenor, pillar, ticker in tickers:
                        if ticker == security:
                            field_data = sec.getElement("fieldData")
                            if field_data.hasElement("PX_LAST"):
                                vol = field_data.getElementAsFloat("PX_LAST")

                                if tenor not in vol_surface:
                                    vol_surface[tenor] = {}
                                vol_surface[tenor][pillar] = vol
                            break

        except Exception as e:
            logger.error(f"Error parsing volatility surface for {cross}: {e}")

        return vol_surface

    def get_all_market_data(self, cross: str, force_refresh: bool = False) -> FXMarketData:
        """
        Fetch all market data for an FX cross.

        Args:
            cross: FX pair (e.g., 'EURUSD')
            force_refresh: If True, bypass cache

        Returns:
            FXMarketData object with all data
        """
        if not force_refresh and cross in self._market_data_cache:
            cached = self._market_data_cache[cross]
            # Check if cache is still fresh (within refresh interval)
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age < self.config.refresh_interval:
                return cached

        market_data = FXMarketData(cross=cross)

        # Fetch spot
        spot = self.get_spot_rate(cross)
        if spot:
            market_data.spot = spot

        # Fetch forward points
        fwd_points = self.get_forward_points(cross)
        market_data.forward_points = fwd_points

        # Calculate forward rates from points
        if spot and fwd_points:
            # Forward points are typically in pips
            pip_factor = 10000 if 'JPY' not in cross else 100
            for tenor, points in fwd_points.items():
                market_data.forward_rates[tenor] = spot + (points / pip_factor)

        # Fetch vol surface
        vol_surface = self.get_volatility_surface(cross)
        market_data.vol_surface = vol_surface

        market_data.timestamp = datetime.now()
        self._market_data_cache[cross] = market_data

        return market_data

    def refresh_all_crosses(self, crosses: List[str] = None) -> Dict[str, FXMarketData]:
        """Refresh market data for all configured crosses."""
        if crosses is None:
            crosses = self.config.fx_crosses

        result = {}
        for cross in crosses:
            result[cross] = self.get_all_market_data(cross, force_refresh=True)

        return result

    def get_forward_rate_for_date(self, cross: str, value_date: date) -> Optional[float]:
        """
        Get interpolated forward rate for a specific value date.

        Uses linear interpolation between available tenor points.
        """
        market_data = self.get_all_market_data(cross)

        if not market_data.spot or not market_data.forward_points:
            return None

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

        # Find surrounding tenors for interpolation
        pip_factor = 10000 if 'JPY' not in cross else 100

        # If before first tenor, use first tenor
        if days_to_value <= tenor_days[0][0]:
            ratio = days_to_value / tenor_days[0][0]
            interp_points = tenor_days[0][1] * ratio
            return market_data.spot + (interp_points / pip_factor)

        # If after last tenor, extrapolate
        if days_to_value >= tenor_days[-1][0]:
            # Linear extrapolation from last two points
            if len(tenor_days) >= 2:
                d1, p1 = tenor_days[-2]
                d2, p2 = tenor_days[-1]
                slope = (p2 - p1) / (d2 - d1)
                interp_points = p2 + slope * (days_to_value - d2)
            else:
                interp_points = tenor_days[-1][1]
            return market_data.spot + (interp_points / pip_factor)

        # Linear interpolation between two points
        for i in range(len(tenor_days) - 1):
            d1, p1 = tenor_days[i]
            d2, p2 = tenor_days[i + 1]

            if d1 <= days_to_value <= d2:
                ratio = (days_to_value - d1) / (d2 - d1)
                interp_points = p1 + ratio * (p2 - p1)
                return market_data.spot + (interp_points / pip_factor)

        return market_data.spot


# Singleton instance
_bloomberg_client = None


def get_bloomberg_client() -> BloombergClient:
    """Get the singleton Bloomberg client instance."""
    global _bloomberg_client
    if _bloomberg_client is None:
        _bloomberg_client = BloombergClient()
    return _bloomberg_client
