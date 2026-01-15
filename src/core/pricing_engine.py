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
    """Represents an FX option position.

    Price convention: All prices are expressed as % of notional in base currency.
    For example, a EURUSD call with price 0.0125 means 1.25% of EUR notional.
    For USDJPY, price 0.85 means 0.85% of USD notional.
    """
    trade_date: date
    cross: str
    expiry: date
    direction: str  # 'Long' or 'Short'
    option_type: str  # 'Call' or 'Put'
    strike: float
    notional: float  # In base currency (e.g., EUR for EURUSD, USD for USDJPY)
    trade_price: float  # Premium as % of notional (e.g., 0.0125 = 1.25%)

    # Calculated fields
    current_price: float = 0.0  # Current premium as % of notional
    delta: float = 0.0  # Forward delta (0-1 for calls, -1-0 for puts)
    delta_notional: float = 0.0  # Delta * signed notional
    vega_usd: float = 0.0  # Vega in USD
    vega_eur: float = 0.0  # Vega in EUR
    gamma_1pct: float = 0.0  # Delta change for 1% spot move
    pnl: float = 0.0  # P&L in EUR

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

        The returned price is expressed as % of notional (e.g., 0.0125 = 1.25%).

        Args:
            option: FXOption object
            spot: Current spot rate
            forward: Forward rate for expiry
            volatility: Implied volatility in % (e.g., 10 for 10%)
            domestic_rate: Domestic risk-free rate (annual)
            foreign_rate: Foreign risk-free rate (annual)

        Returns:
            OptionGreeks object with price (as % of notional) and Greeks
        """
        greeks = OptionGreeks()

        try:
            # Set up dates
            valuation_date = DateUtils.to_ql_date(date.today())
            expiry_date = DateUtils.to_ql_date(option.expiry)

            # Check if expired
            if expiry_date <= valuation_date:
                # Option expired - calculate intrinsic value as % of notional
                if option.is_call:
                    intrinsic = max(0, spot - option.strike)
                else:
                    intrinsic = max(0, option.strike - spot)
                # Convert to % of notional: intrinsic / spot for non-JPY
                if 'JPY' in option.cross:
                    greeks.price = intrinsic / spot  # % of notional
                else:
                    greeks.price = intrinsic / spot if spot > 0 else 0
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
            # QuantLib returns price in quote currency per unit of base
            # We need to convert to % of notional
            raw_price = european_option.NPV()

            logger.debug(f"Pricing {option.cross} {option.option_type} K={option.strike:.5f}: "
                        f"S={spot:.5f}, F={forward:.5f}, vol={volatility:.2f}%, T={time_to_expiry:.4f}y, "
                        f"raw_price={raw_price:.6f}")

            # Convert price to % of notional
            # For FX options, premium is typically quoted as % of base currency notional
            # QuantLib gives price in domestic (quote) currency per 1 unit of foreign (base)
            # To get %, we divide by spot
            if spot > 0:
                greeks.price = raw_price / spot
            else:
                greeks.price = 0.0

            logger.debug(f"  price_pct={greeks.price:.6f} ({greeks.price*100:.4f}%)")

            greeks.delta = european_option.delta()
            greeks.gamma = european_option.gamma()
            greeks.vega = european_option.vega() / 100  # Per 1% vol move
            greeks.theta = european_option.theta() / 365  # Per day

            logger.debug(f"  delta={greeks.delta:.4f}, gamma={greeks.gamma:.6f}, vega={greeks.vega:.4f}")

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
                      volatility: float, notional: float,
                      eurusd: float = None) -> Tuple[float, float]:
        """
        Calculate vega in USD and EUR terms.

        Vega is the change in option value (in base currency) for 1% vol change.

        Args:
            option: FXOption object
            spot: Spot rate
            forward: Forward rate
            volatility: Implied volatility in %
            notional: Option notional
            eurusd: EURUSD rate for conversion to EUR

        Returns:
            Tuple of (vega_usd, vega_eur)
        """
        try:
            greeks = self.price_option(option, spot, forward, volatility)

            # Vega from greeks is in % terms (per notional unit)
            # Multiply by notional to get vega in base currency
            vega_base = greeks.vega * abs(notional) * spot  # Convert back to currency

            # Convert to USD and EUR based on cross
            cross = option.cross.upper()

            if eurusd is None or eurusd <= 0:
                eurusd = 1.08  # Default fallback

            if cross.startswith('EUR'):
                # Base currency is EUR
                vega_eur = vega_base
                if 'USD' in cross:
                    vega_usd = vega_base * spot  # EUR to USD
                else:
                    vega_usd = vega_base * eurusd  # Approximate
            elif cross.startswith('USD'):
                # Base currency is USD (e.g., USDJPY, USDCAD)
                vega_usd = vega_base
                vega_eur = vega_base / eurusd
            elif cross.endswith('USD'):
                # Quote currency is USD (e.g., AUDUSD, GBPUSD)
                # Vega is in base currency (AUD, GBP)
                vega_usd = vega_base * spot  # Convert base to USD
                vega_eur = vega_usd / eurusd
            else:
                # Other crosses (e.g., EURGBP, AUDNZD)
                vega_usd = vega_base
                vega_eur = vega_base

            # Apply direction sign
            sign = 1 if option.is_long else -1

            return vega_usd * sign, vega_eur * sign

        except Exception as e:
            logger.error(f"Error calculating vega: {e}")
            return 0.0, 0.0

    def calculate_pnl(self, option: FXOption, current_price: float,
                      spot: float = None, eurusd: float = None) -> float:
        """
        Calculate P&L for an option position in EUR.

        Prices are expressed as % of notional in base currency.
        P&L is converted to EUR for aggregation.

        Args:
            option: FXOption object with trade_price (as % of notional)
            current_price: Current option price (as % of notional)
            spot: Current spot rate for the cross
            eurusd: EURUSD rate for conversion to EUR

        Returns:
            P&L in EUR
        """
        try:
            # Price difference (both are % of notional)
            price_diff = current_price - option.trade_price

            # Apply direction
            sign = 1 if option.is_long else -1

            # P&L in base currency = notional * price_diff * sign
            pnl_base = option.notional * price_diff * sign

            # Convert to EUR based on cross
            cross = option.cross.upper()

            if cross.startswith('EUR'):
                # Base currency is EUR, P&L already in EUR
                pnl_eur = pnl_base
            elif cross.startswith('USD'):
                # Base currency is USD, convert to EUR
                if eurusd and eurusd > 0:
                    pnl_eur = pnl_base / eurusd
                else:
                    pnl_eur = pnl_base  # Fallback
            elif cross.endswith('USD'):
                # Quote currency is USD (e.g., AUDUSD, GBPUSD)
                # Base is AUD/GBP, need to convert via USD then EUR
                if spot and spot > 0 and eurusd and eurusd > 0:
                    pnl_usd = pnl_base * spot
                    pnl_eur = pnl_usd / eurusd
                else:
                    pnl_eur = pnl_base
            else:
                # Other crosses - approximate
                pnl_eur = pnl_base

            return pnl_eur

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
        # Get EURUSD rate for currency conversion
        eurusd = None
        if 'EURUSD' in market_data:
            eurusd = market_data['EURUSD'].spot

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

            logger.info(f"Pricing {cross} {option.option_type} K={option.strike:.5f} exp={option.expiry}: "
                       f"S={md.spot:.5f}, F={forward:.5f}, vol={volatility:.2f}%, T={time_to_expiry:.4f}y")

            # Price option
            greeks = self.price_option(option, md.spot, forward, volatility)

            # Calculate forward delta
            delta = self.calculate_forward_delta(option, md.spot, forward, volatility)

            # Calculate gamma 1%
            gamma_1pct = self.calculate_gamma_1pct(option, md.spot, forward, volatility, vol_surface)

            # Calculate vega
            vega_usd, vega_eur = self.calculate_vega(
                option, md.spot, forward, volatility, option.notional, eurusd
            )

            # Update option with calculated values
            option.current_price = greeks.price
            option.delta = delta
            option.delta_notional = delta * option.signed_notional
            option.gamma_1pct = gamma_1pct * option.signed_notional
            option.vega_usd = vega_usd
            option.vega_eur = vega_eur

            # Calculate P&L
            option.pnl = self.calculate_pnl(option, greeks.price, md.spot, eurusd)

        return options
