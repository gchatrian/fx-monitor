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
            logger.debug("Bloomberg: Already connected, reusing existing session")
            return True

        logger.info(f"Bloomberg: Attempting connection to {self.config.bloomberg_host}:{self.config.bloomberg_port}")

        try:
            session_options = blpapi.SessionOptions()
            session_options.setServerHost(self.config.bloomberg_host)
            session_options.setServerPort(self.config.bloomberg_port)

            logger.debug("Bloomberg: Creating session with options")
            self.session = blpapi.Session(session_options)

            logger.debug("Bloomberg: Starting session...")
            if not self.session.start():
                logger.error("Bloomberg: Failed to start session - is Bloomberg Terminal running?")
                return False

            logger.debug("Bloomberg: Session started, opening //blp/refdata service...")
            if not self.session.openService("//blp/refdata"):
                logger.error("Bloomberg: Failed to open //blp/refdata service")
                return False

            self.ref_data_service = self.session.getService("//blp/refdata")
            self._connected = True
            logger.info("Bloomberg: Successfully connected to Bloomberg API")
            return True

        except Exception as e:
            logger.error(f"Bloomberg: Connection error - {type(e).__name__}: {e}")
            return False

    def disconnect(self) -> None:
        """Close Bloomberg session."""
        if self.session:
            logger.info("Bloomberg: Disconnecting from API...")
            self.session.stop()
            self._connected = False
            logger.info("Bloomberg: Disconnected successfully")

    def is_connected(self) -> bool:
        """Check if connected to Bloomberg."""
        return self._connected

    def _send_request(self, request, request_type: str = "Unknown") -> Optional[List[blpapi.Message]]:
        """Send a request and collect responses."""
        if not self._connected:
            logger.warning(f"Bloomberg: Not connected, attempting reconnection for {request_type} request")
            if not self.connect():
                logger.error(f"Bloomberg: Reconnection failed, cannot execute {request_type} request")
                return None

        try:
            logger.debug(f"Bloomberg: Sending {request_type} request...")
            self.session.sendRequest(request)
            messages = []

            event_count = 0
            while True:
                event = self.session.nextEvent(self.config.bloomberg_timeout)
                event_count += 1

                for msg in event:
                    messages.append(msg)
                    # Log any errors in the response
                    if msg.hasElement("responseError"):
                        error = msg.getElement("responseError")
                        logger.error(f"Bloomberg: Response error in {request_type}: {error}")

                if event.eventType() == blpapi.Event.RESPONSE:
                    logger.debug(f"Bloomberg: {request_type} request completed after {event_count} events, {len(messages)} messages received")
                    break

            return messages

        except blpapi.exception.InvalidStateException as e:
            logger.error(f"Bloomberg: Invalid state for {request_type} request - {e}")
            self._connected = False
            return None
        except Exception as e:
            logger.error(f"Bloomberg: Request error for {request_type} - {type(e).__name__}: {e}")
            return None

    def get_spot_rate(self, cross: str) -> Optional[float]:
        """Fetch spot rate for an FX cross."""
        ticker = f"{cross} Curncy"
        logger.info(f"Bloomberg: Fetching spot rate for {cross} (ticker: {ticker})")

        request = self.ref_data_service.createRequest("ReferenceDataRequest")
        request.append("securities", ticker)
        request.append("fields", "PX_LAST")

        messages = self._send_request(request, f"Spot {cross}")
        if not messages:
            logger.warning(f"Bloomberg: No response received for spot rate {cross}")
            return None

        try:
            for msg in messages:
                if not msg.hasElement("securityData"):
                    continue

                security_data = msg.getElement("securityData")
                for i in range(security_data.numValues()):
                    sec = security_data.getValueAsElement(i)

                    # Check for security-level errors
                    if sec.hasElement("securityError"):
                        error = sec.getElement("securityError")
                        logger.error(f"Bloomberg: Security error for {ticker}: {error}")
                        continue

                    field_data = sec.getElement("fieldData")
                    if field_data.hasElement("PX_LAST"):
                        spot = field_data.getElementAsFloat("PX_LAST")
                        logger.info(f"Bloomberg: {cross} spot rate = {spot:.5f}")
                        return spot
                    else:
                        logger.warning(f"Bloomberg: PX_LAST field not found for {ticker}")

        except Exception as e:
            logger.error(f"Bloomberg: Error parsing spot rate for {cross} - {type(e).__name__}: {e}")

        logger.warning(f"Bloomberg: Could not retrieve spot rate for {cross}")
        return None

    def get_forward_points(self, cross: str) -> Dict[str, float]:
        """Fetch forward points curve for an FX cross."""
        forward_points = {}
        tenors = self.config.forward_tenors

        logger.info(f"Bloomberg: Fetching forward points for {cross} ({len(tenors)} tenors)")

        # Build tickers for each tenor
        tickers = []
        for tenor in tenors:
            # Bloomberg convention: EURUSD1M Curncy for 1M forward points
            ticker = f"{cross}{tenor} Curncy"
            tickers.append((tenor, ticker))

        logger.debug(f"Bloomberg: Forward tickers to request: {[t[1] for t in tickers]}")

        request = self.ref_data_service.createRequest("ReferenceDataRequest")
        for _, ticker in tickers:
            request.append("securities", ticker)
        request.append("fields", "PX_LAST")

        messages = self._send_request(request, f"Forward Points {cross}")
        if not messages:
            logger.warning(f"Bloomberg: No response received for forward points {cross}")
            return forward_points

        try:
            for msg in messages:
                if not msg.hasElement("securityData"):
                    continue

                security_data = msg.getElement("securityData")
                for i in range(security_data.numValues()):
                    sec = security_data.getValueAsElement(i)
                    security = sec.getElementAsString("security")

                    # Check for security-level errors
                    if sec.hasElement("securityError"):
                        error = sec.getElement("securityError")
                        logger.warning(f"Bloomberg: Security error for {security}: {error}")
                        continue

                    # Extract tenor from ticker
                    for tenor, ticker in tickers:
                        if ticker == security:
                            field_data = sec.getElement("fieldData")
                            if field_data.hasElement("PX_LAST"):
                                points = field_data.getElementAsFloat("PX_LAST")
                                forward_points[tenor] = points
                                logger.debug(f"Bloomberg: {cross} {tenor} fwd points = {points:.2f}")
                            else:
                                logger.warning(f"Bloomberg: PX_LAST not found for {ticker}")
                            break

            logger.info(f"Bloomberg: Retrieved {len(forward_points)}/{len(tenors)} forward points for {cross}")
            if forward_points:
                logger.debug(f"Bloomberg: {cross} forward points: {forward_points}")

        except Exception as e:
            logger.error(f"Bloomberg: Error parsing forward points for {cross} - {type(e).__name__}: {e}")

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

        total_points = len(tenors) * len(pillars)
        logger.info(f"Bloomberg: Fetching volatility surface for {cross} ({len(tenors)} tenors x {len(pillars)} pillars = {total_points} points)")

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
                    logger.warning(f"Bloomberg: Unknown vol pillar: {pillar}")
                    continue

                tickers.append((tenor, pillar, ticker))

        logger.debug(f"Bloomberg: Vol surface tickers to request: {len(tickers)} tickers")

        request = self.ref_data_service.createRequest("ReferenceDataRequest")
        for _, _, ticker in tickers:
            request.append("securities", ticker)
        request.append("fields", "PX_LAST")

        messages = self._send_request(request, f"Vol Surface {cross}")
        if not messages:
            logger.warning(f"Bloomberg: No response received for vol surface {cross}")
            return vol_surface

        received_count = 0
        error_count = 0

        try:
            for msg in messages:
                if not msg.hasElement("securityData"):
                    continue

                security_data = msg.getElement("securityData")
                for i in range(security_data.numValues()):
                    sec = security_data.getValueAsElement(i)
                    security = sec.getElementAsString("security")

                    # Check for security-level errors
                    if sec.hasElement("securityError"):
                        error = sec.getElement("securityError")
                        logger.warning(f"Bloomberg: Security error for {security}: {error}")
                        error_count += 1
                        continue

                    for tenor, pillar, ticker in tickers:
                        if ticker == security:
                            field_data = sec.getElement("fieldData")
                            if field_data.hasElement("PX_LAST"):
                                vol = field_data.getElementAsFloat("PX_LAST")

                                if tenor not in vol_surface:
                                    vol_surface[tenor] = {}
                                vol_surface[tenor][pillar] = vol
                                received_count += 1
                                logger.debug(f"Bloomberg: {cross} {tenor} {pillar} = {vol:.2f}%")
                            break

            logger.info(f"Bloomberg: Retrieved {received_count}/{total_points} vol surface points for {cross} ({error_count} errors)")

            # Log summary of vol surface
            if vol_surface:
                for tenor in sorted(vol_surface.keys()):
                    pillars_str = ", ".join([f"{p}={v:.2f}" for p, v in vol_surface[tenor].items()])
                    logger.debug(f"Bloomberg: {cross} {tenor}: {pillars_str}")

        except Exception as e:
            logger.error(f"Bloomberg: Error parsing volatility surface for {cross} - {type(e).__name__}: {e}")

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
        logger.info(f"Bloomberg: {'Force refreshing' if force_refresh else 'Fetching'} all market data for {cross}")

        if not force_refresh and cross in self._market_data_cache:
            cached = self._market_data_cache[cross]
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age < self.config.refresh_interval:
                logger.debug(f"Bloomberg: Using cached data for {cross} (age: {age:.1f}s)")
                return cached

        market_data = FXMarketData(cross=cross)
        start_time = datetime.now()

        # Fetch spot
        logger.debug(f"Bloomberg: Step 1/3 - Fetching spot for {cross}")
        spot = self.get_spot_rate(cross)
        if spot:
            market_data.spot = spot
        else:
            logger.warning(f"Bloomberg: No spot rate received for {cross}")

        # Fetch forward points
        logger.debug(f"Bloomberg: Step 2/3 - Fetching forward points for {cross}")
        fwd_points = self.get_forward_points(cross)
        market_data.forward_points = fwd_points

        # Calculate forward rates from points
        if spot and fwd_points:
            pip_factor = 10000 if 'JPY' not in cross else 100
            for tenor, points in fwd_points.items():
                market_data.forward_rates[tenor] = spot + (points / pip_factor)
            logger.debug(f"Bloomberg: Calculated {len(market_data.forward_rates)} forward rates for {cross}")

        # Fetch vol surface
        logger.debug(f"Bloomberg: Step 3/3 - Fetching vol surface for {cross}")
        vol_surface = self.get_volatility_surface(cross)
        market_data.vol_surface = vol_surface

        market_data.timestamp = datetime.now()
        self._market_data_cache[cross] = market_data

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Bloomberg: Completed market data fetch for {cross} in {elapsed:.2f}s - "
                   f"Spot: {market_data.spot:.5f}, "
                   f"Fwd Points: {len(market_data.forward_points)}, "
                   f"Vol Tenors: {len(market_data.vol_surface)}")

        return market_data

    def refresh_all_crosses(self, crosses: List[str] = None) -> Dict[str, FXMarketData]:
        """Refresh market data for all configured crosses."""
        if crosses is None:
            crosses = self.config.fx_crosses

        logger.info(f"Bloomberg: Refreshing market data for {len(crosses)} crosses: {crosses}")
        start_time = datetime.now()

        result = {}
        for i, cross in enumerate(crosses, 1):
            logger.info(f"Bloomberg: Processing cross {i}/{len(crosses)}: {cross}")
            result[cross] = self.get_all_market_data(cross, force_refresh=True)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Bloomberg: Completed refresh of all {len(crosses)} crosses in {elapsed:.2f}s")

        return result

    def get_forward_rate_for_date(self, cross: str, value_date: date) -> Optional[float]:
        """
        Get interpolated forward rate for a specific value date.

        Uses linear interpolation between available tenor points.
        """
        logger.debug(f"Bloomberg: Calculating forward rate for {cross} value date {value_date}")

        market_data = self.get_all_market_data(cross)

        if not market_data.spot or not market_data.forward_points:
            logger.warning(f"Bloomberg: Cannot calculate forward rate for {cross} - missing spot or forward points")
            return None

        today = date.today()
        days_to_value = (value_date - today).days

        if days_to_value <= 0:
            logger.debug(f"Bloomberg: Value date {value_date} is today or in the past, returning spot {market_data.spot}")
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
            logger.warning(f"Bloomberg: No valid tenor points for interpolation for {cross}")
            return market_data.spot

        pip_factor = 10000 if 'JPY' not in cross else 100

        # If before first tenor, use first tenor
        if days_to_value <= tenor_days[0][0]:
            ratio = days_to_value / tenor_days[0][0]
            interp_points = tenor_days[0][1] * ratio
            fwd_rate = market_data.spot + (interp_points / pip_factor)
            logger.debug(f"Bloomberg: {cross} {value_date} ({days_to_value}d) - interpolated before first tenor: {fwd_rate:.5f}")
            return fwd_rate

        # If after last tenor, extrapolate
        if days_to_value >= tenor_days[-1][0]:
            if len(tenor_days) >= 2:
                d1, p1 = tenor_days[-2]
                d2, p2 = tenor_days[-1]
                slope = (p2 - p1) / (d2 - d1)
                interp_points = p2 + slope * (days_to_value - d2)
            else:
                interp_points = tenor_days[-1][1]
            fwd_rate = market_data.spot + (interp_points / pip_factor)
            logger.debug(f"Bloomberg: {cross} {value_date} ({days_to_value}d) - extrapolated beyond last tenor: {fwd_rate:.5f}")
            return fwd_rate

        # Linear interpolation between two points
        for i in range(len(tenor_days) - 1):
            d1, p1 = tenor_days[i]
            d2, p2 = tenor_days[i + 1]

            if d1 <= days_to_value <= d2:
                ratio = (days_to_value - d1) / (d2 - d1)
                interp_points = p1 + ratio * (p2 - p1)
                fwd_rate = market_data.spot + (interp_points / pip_factor)
                logger.debug(f"Bloomberg: {cross} {value_date} ({days_to_value}d) - interpolated between {d1}d and {d2}d: {fwd_rate:.5f}")
                return fwd_rate

        return market_data.spot


# Singleton instance
_bloomberg_client = None


def get_bloomberg_client() -> BloombergClient:
    """Get the singleton Bloomberg client instance."""
    global _bloomberg_client
    if _bloomberg_client is None:
        _bloomberg_client = BloombergClient()
    return _bloomberg_client
