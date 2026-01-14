"""QuantLib-based pricing engine for FX options."""

import QuantLib as ql
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Dict, Tuple
import math
import logging

from ..utils.date_utils import DateUtils
from .volatility_surface import VolatilitySurface

logger = logging.getLogger(__name__)


@dataclass
class FXOption:
    """Represents an FX option position."""
    trade_date: date
    cross: str
    expiry: date
    direction: str  # 'Long' or 'Short'
    option_type: str  # 'Call' or 'Put'
    strike: float
    notional: float  # In base currency
    trade_price: float  # Premium paid/received in pips or %

    # Calculated fields
    current_price: float = 0.0
    delta: float = 0.0
    delta_notional: float = 0.0
    vega_usd: float = 0.0
    vega_eur: float = 0.0
    gamma_1pct: float = 0.0
    pnl: float = 0.0

    def __post_init__(self):
        """Normalize input values."""
        self.direction = self.direction.capitalize()
        self.option_type = self.option_type.capitalize()

        # Convert string dates if necessary
        if isinstance(self.trade_date, str):
            self.trade_date = DateUtils.from_ql_date(DateUtils.to_ql_date(self.trade_date))
        if isinstance(self.expiry, str):
            self.expiry = DateUtils.from_ql_date(DateUtils.to_ql_date(self.expiry))

    @property
    def is_call(self) -> bool:
        return self.option_type.lower() == 'call'

    @property
    def is_long(self) -> bool:
        return self.direction.lower() == 'long'

    @property
    def signed_notional(self) -> float:
        """Notional with sign based on direction."""
        return self.notional if self.is_long else -self.notional

    @property
    def expiry_str(self) -> str:
        """Return expiry as string for grouping."""
        return self.expiry.strftime('%Y-%m-%d')


@dataclass
class OptionGreeks:
    """Container for option Greeks."""
    price: float = 0.0  # Option premium
    delta: float = 0.0  # Forward delta
    gamma: float = 0.0  # Gamma
    vega: float = 0.0   # Vega per 1% vol move
    theta: float = 0.0  # Theta per day
    rho: float = 0.0    # Rho


class FXOptionPricer:
    """
    Pricing engine for FX vanilla options using QuantLib.

    Uses Garman-Kohlhagen model (Black-Scholes for FX).
    """

    def __init__(self):
        self.day_count = ql.Actual365Fixed()
        self.calendar = ql.TARGET()

    def _setup_market_environment(self, valuation_date: ql.Date, spot: float,
                                  domestic_rate: float, foreign_rate: float,
                                  volatility: float) -> Tuple:
        """
        Set up QuantLib market environment.

        Args:
            valuation_date: Valuation date
            spot: Spot rate
            domestic_rate: Domestic risk-free rate (quote currency)
            foreign_rate: Foreign risk-free rate (base currency)
            volatility: Implied volatility (decimal)

        Returns:
            Tuple of (spot_handle, rate_ts_dom, rate_ts_for, vol_ts)
        """
        ql.Settings.instance().evaluationDate = valuation_date

        # Spot handle
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))

        # Flat rate curves
        rate_ts_dom = ql.YieldTermStructureHandle(
            ql.FlatForward(valuation_date, domestic_rate, self.day_count)
        )
        rate_ts_for = ql.YieldTermStructureHandle(
            ql.FlatForward(valuation_date, foreign_rate, self.day_count)
        )

        # Flat volatility surface
        vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(valuation_date, self.calendar, volatility, self.day_count)
        )

        return spot_handle, rate_ts_dom, rate_ts_for, vol_ts

    def price_option(self, option: FXOption, spot: float, forward: float,
                     volatility: float, domestic_rate: float = 0.0,
                     foreign_rate: float = 0.0) -> OptionGreeks:
        """
        Price an FX option using QuantLib.

        Args:
            option: FXOption object
            spot: Current spot rate
            forward: Forward rate for expiry
            volatility: Implied volatility in % (e.g., 10 for 10%)
            domestic_rate: Domestic risk-free rate (annual)
            foreign_rate: Foreign risk-free rate (annual)

        Returns:
            OptionGreeks object with price and Greeks
        """
        greeks = OptionGreeks()

        try:
            # Set up dates
            valuation_date = DateUtils.to_ql_date(date.today())
            expiry_date = DateUtils.to_ql_date(option.expiry)

            # Check if expired
            if expiry_date <= valuation_date:
                # Option expired - calculate intrinsic value only
                if option.is_call:
                    intrinsic = max(0, spot - option.strike)
                else:
                    intrinsic = max(0, option.strike - spot)
                greeks.price = intrinsic
                return greeks

            # If rates not provided, derive from forward
            time_to_expiry = self.day_count.yearFraction(valuation_date, expiry_date)
            if time_to_expiry > 0:
                # Forward = Spot * exp((rd - rf) * T)
                # => rd - rf = ln(Forward/Spot) / T
                rate_diff = math.log(forward / spot) / time_to_expiry if spot > 0 else 0

                if domestic_rate == 0 and foreign_rate == 0:
                    # Assume symmetric rates around the difference
                    domestic_rate = rate_diff / 2
                    foreign_rate = -rate_diff / 2

            # Convert volatility to decimal
            vol_decimal = volatility / 100

            # Set up market environment
            spot_handle, rate_ts_dom, rate_ts_for, vol_ts = self._setup_market_environment(
                valuation_date, spot, domestic_rate, foreign_rate, vol_decimal
            )

            # Create the option
            option_type = ql.Option.Call if option.is_call else ql.Option.Put
            payoff = ql.PlainVanillaPayoff(option_type, option.strike)
            exercise = ql.EuropeanExercise(expiry_date)
            european_option = ql.VanillaOption(payoff, exercise)

            # Create Garman-Kohlhagen process
            gk_process = ql.GarmanKohlagenProcess(
                spot_handle, rate_ts_for, rate_ts_dom, vol_ts
            )

            # Use analytic European engine
            engine = ql.AnalyticEuropeanEngine(gk_process)
            european_option.setPricingEngine(engine)

            # Calculate price and Greeks
            greeks.price = european_option.NPV()
            greeks.delta = european_option.delta()
            greeks.gamma = european_option.gamma()
            greeks.vega = european_option.vega() / 100  # Per 1% vol move
            greeks.theta = european_option.theta() / 365  # Per day

            try:
                greeks.rho = european_option.rho()
            except:
                greeks.rho = 0.0

        except Exception as e:
            logger.error(f"Error pricing option: {e}")

        return greeks

    def calculate_forward_delta(self, option: FXOption, spot: float, forward: float,
                                volatility: float) -> float:
        """
        Calculate forward delta using Black-Scholes.

        Forward delta is the sensitivity to the forward rate, not the spot.
        """
        try:
            valuation_date = DateUtils.to_ql_date(date.today())
            expiry_date = DateUtils.to_ql_date(option.expiry)

            if expiry_date <= valuation_date:
                # Expired - delta is 1 or 0 based on moneyness
                if option.is_call:
                    return 1.0 if spot > option.strike else 0.0
                else:
                    return -1.0 if spot < option.strike else 0.0

            time_to_expiry = self.day_count.yearFraction(valuation_date, expiry_date)
            if time_to_expiry <= 0:
                time_to_expiry = 1/365

            vol_decimal = volatility / 100
            sqrt_t = math.sqrt(time_to_expiry)

            from scipy.stats import norm

            # d1 = (ln(F/K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
            d1 = (math.log(forward / option.strike) + 0.5 * vol_decimal**2 * time_to_expiry) / (vol_decimal * sqrt_t)

            if option.is_call:
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1

            return delta

        except Exception as e:
            logger.error(f"Error calculating forward delta: {e}")
            return 0.0

    def calculate_gamma_1pct(self, option: FXOption, spot: float, forward: float,
                            volatility: float, vol_surface: VolatilitySurface = None) -> float:
        """
        Calculate gamma as change in delta for 1% spot move.

        This is a practical measure of gamma in delta terms.
        """
        try:
            # Calculate delta at current spot
            delta_0 = self.calculate_forward_delta(option, spot, forward, volatility)

            # Calculate delta at +1% spot
            spot_up = spot * 1.01
            forward_up = forward * 1.01  # Assume parallel shift

            # If vol surface provided, get new vol for shifted spot
            if vol_surface:
                valuation_date = date.today()
                expiry_date = option.expiry
                time_to_expiry = (expiry_date - valuation_date).days / 365
                volatility_up = vol_surface.get_vol_for_strike(time_to_expiry, option.strike, forward_up)
            else:
                volatility_up = volatility

            delta_up = self.calculate_forward_delta(option, spot_up, forward_up, volatility_up)

            # Gamma 1% = delta change for 1% spot move
            gamma_1pct = delta_up - delta_0

            return gamma_1pct

        except Exception as e:
            logger.error(f"Error calculating gamma 1%: {e}")
            return 0.0

    def calculate_vega(self, option: FXOption, spot: float, forward: float,
                      volatility: float, notional: float) -> Tuple[float, float]:
        """
        Calculate vega in USD and EUR terms.

        Returns:
            Tuple of (vega_usd, vega_eur)
        """
        try:
            greeks = self.price_option(option, spot, forward, volatility)

            # Vega is price sensitivity to 1% vol change
            # greeks.vega is per notional unit
            vega_base = greeks.vega * abs(notional)

            # Convert to USD and EUR based on cross
            cross = option.cross.upper()

            if cross.startswith('EUR'):
                # Base currency is EUR
                vega_eur = vega_base
                vega_usd = vega_base * spot if 'USD' in cross else vega_base
            elif cross.endswith('USD'):
                # Quote currency is USD
                vega_usd = vega_base * spot
                vega_eur = vega_usd / spot if spot > 0 else 0
            else:
                # Other cross - use notional as base
                vega_usd = vega_base
                vega_eur = vega_base

            # Apply direction sign
            sign = 1 if option.is_long else -1

            return vega_usd * sign, vega_eur * sign

        except Exception as e:
            logger.error(f"Error calculating vega: {e}")
            return 0.0, 0.0

    def calculate_pnl(self, option: FXOption, current_price: float) -> float:
        """
        Calculate P&L for an option position.

        Args:
            option: FXOption object with trade_price
            current_price: Current option price

        Returns:
            P&L in quote currency terms
        """
        try:
            # Price difference
            price_diff = current_price - option.trade_price

            # Apply direction
            sign = 1 if option.is_long else -1

            # P&L = notional * price_diff * sign
            # Assuming prices are in pips
            pip_factor = 10000 if 'JPY' not in option.cross else 100

            pnl = option.notional * price_diff * sign / pip_factor

            return pnl

        except Exception as e:
            logger.error(f"Error calculating P&L: {e}")
            return 0.0

    def price_portfolio(self, options: list, market_data: Dict,
                       vol_surfaces: Dict[str, VolatilitySurface]) -> list:
        """
        Price a portfolio of options.

        Args:
            options: List of FXOption objects
            market_data: Dict of {cross: FXMarketData}
            vol_surfaces: Dict of {cross: VolatilitySurface}

        Returns:
            List of FXOption objects with calculated Greeks
        """
        for option in options:
            cross = option.cross.upper()

            if cross not in market_data:
                logger.warning(f"No market data for {cross}")
                continue

            md = market_data[cross]
            vol_surface = vol_surfaces.get(cross)

            # Calculate time to expiry
            today = date.today()
            time_to_expiry = (option.expiry - today).days / 365

            if time_to_expiry <= 0:
                # Expired option
                continue

            # Get forward rate for expiry
            if vol_surface:
                forward = vol_surface.get_forward_for_expiry(time_to_expiry)
            else:
                forward = md.spot

            # Get volatility
            if vol_surface:
                volatility = vol_surface.get_vol_for_strike(time_to_expiry, option.strike, forward)
            else:
                volatility = 10.0  # Default

            # Price option
            greeks = self.price_option(option, md.spot, forward, volatility)

            # Calculate forward delta
            delta = self.calculate_forward_delta(option, md.spot, forward, volatility)

            # Calculate gamma 1%
            gamma_1pct = self.calculate_gamma_1pct(option, md.spot, forward, volatility, vol_surface)

            # Calculate vega
            vega_usd, vega_eur = self.calculate_vega(option, md.spot, forward, volatility, option.notional)

            # Update option with calculated values
            option.current_price = greeks.price
            option.delta = delta
            option.delta_notional = delta * option.signed_notional
            option.gamma_1pct = gamma_1pct * option.signed_notional
            option.vega_usd = vega_usd
            option.vega_eur = vega_eur

            # Calculate P&L
            option.pnl = self.calculate_pnl(option, greeks.price)

        return options
