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
                volatility_up = vol_surface.get_vol_for_strike(time_to_expiry, option.strike, forward_up, option.is_call)
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
                      market_data: Dict = None) -> Tuple[float, float]:
        """
        Calculate vega in USD and EUR terms.

        QuantLib's vega for FX options is in quote currency (e.g., USD for EURUSD).
        We convert to EUR using the appropriate cross rate.

        Args:
            option: FXOption object
            spot: Spot rate
            forward: Forward rate
            volatility: Implied volatility in %
            notional: Option notional
            market_data: Dict of {cross: FXMarketData} for EUR conversion

        Returns:
            Tuple of (vega_usd, vega_eur)
        """
        try:
            greeks = self.price_option(option, spot, forward, volatility)

            cross = option.cross.upper()
            base_ccy = cross[:3]
            quote_ccy = cross[3:]

            # greeks.vega from QuantLib is per unit notional in quote currency
            # (e.g., for EURUSD, vega is in USD per 1 EUR notional)
            # To get total vega in quote currency: vega * notional
            vega_quote = greeks.vega * abs(notional)

            logger.debug(f"Vega calc for {cross}: greeks.vega={greeks.vega:.6f}, "
                        f"notional={notional:,.0f}, vega_quote={vega_quote:,.0f} {quote_ccy}")

            # Get EURUSD for conversions
            eurusd = 1.08  # Default
            if market_data and 'EURUSD' in market_data:
                eurusd = market_data['EURUSD'].spot

            # Convert vega from quote currency to EUR and USD
            if quote_ccy == 'USD':
                # Vega is already in USD
                vega_usd = vega_quote
                vega_eur = vega_quote / eurusd
            elif quote_ccy == 'EUR':
                # Vega is already in EUR (rare case like USDEUR)
                vega_eur = vega_quote
                vega_usd = vega_quote * eurusd
            else:
                # Quote currency is something else (e.g., JPY for USDJPY, GBP for EURGBP)
                # First convert to USD, then to EUR
                # Get USD cross for quote currency
                if quote_ccy == 'JPY':
                    # USDJPY: 1 USD = X JPY, so JPY to USD = amount / USDJPY
                    usdjpy = 150.0  # Default
                    if market_data and 'USDJPY' in market_data:
                        usdjpy = market_data['USDJPY'].spot
                    vega_usd = vega_quote / usdjpy
                elif quote_ccy == 'CAD':
                    usdcad = 1.35  # Default
                    if market_data and 'USDCAD' in market_data:
                        usdcad = market_data['USDCAD'].spot
                    vega_usd = vega_quote / usdcad
                elif quote_ccy == 'CHF':
                    usdchf = 0.90  # Default
                    if market_data and 'USDCHF' in market_data:
                        usdchf = market_data['USDCHF'].spot
                    vega_usd = vega_quote / usdchf
                elif quote_ccy == 'GBP':
                    # GBPUSD: 1 GBP = X USD, so GBP to USD = amount * GBPUSD
                    gbpusd = 1.27  # Default
                    if market_data and 'GBPUSD' in market_data:
                        gbpusd = market_data['GBPUSD'].spot
                    vega_usd = vega_quote * gbpusd
                elif quote_ccy == 'NZD':
                    nzdusd = 0.62  # Default
                    if market_data and 'NZDUSD' in market_data:
                        nzdusd = market_data['NZDUSD'].spot
                    vega_usd = vega_quote * nzdusd
                else:
                    # Fallback: assume quote currency trades vs USD at 1:1
                    vega_usd = vega_quote

                vega_eur = vega_usd / eurusd

            logger.debug(f"  vega_usd={vega_usd:,.0f}, vega_eur={vega_eur:,.0f}")

            # Apply direction sign
            sign = 1 if option.is_long else -1

            return vega_usd * sign, vega_eur * sign

        except Exception as e:
            logger.error(f"Error calculating vega: {e}")
            return 0.0, 0.0

    def calculate_pnl(self, option: FXOption, current_price: float,
                      market_data: Dict = None) -> float:
        """
        Calculate P&L for an option position in EUR.

        Prices are expressed as % of notional in base currency.
        P&L is converted to EUR using direct EUR crosses.

        Args:
            option: FXOption object with trade_price (as % of notional)
            current_price: Current option price (as % of notional)
            market_data: Dict of {cross: FXMarketData} for EUR conversion rates

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

            # Convert to EUR using direct EUR crosses
            cross = option.cross.upper()
            base_ccy = cross[:3]  # First currency (e.g., 'EUR', 'USD', 'GBP', 'AUD')

            if base_ccy == 'EUR':
                # P&L already in EUR
                return pnl_base

            # Get EUR cross for the base currency
            eur_cross = f'EUR{base_ccy}'

            if market_data and eur_cross in market_data:
                eur_rate = market_data[eur_cross].spot
                if eur_rate and eur_rate > 0:
                    # EUR/XXX means 1 EUR = X units of XXX
                    # To convert XXX to EUR: amount / EUR_XXX_rate
                    return pnl_base / eur_rate

            # Fallback to EURUSD if available
            if market_data and 'EURUSD' in market_data:
                eurusd = market_data['EURUSD'].spot
                if eurusd and eurusd > 0:
                    logger.warning(f"No rate for {eur_cross}, using EURUSD fallback")
                    return pnl_base / eurusd

            return pnl_base

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
                volatility = vol_surface.get_vol_for_strike(time_to_expiry, option.strike, forward, option.is_call)
            else:
                volatility = 10.0  # Default

            logger.info(f"Pricing {cross} {option.option_type} K={option.strike:.5f} exp={option.expiry}: "
                       f"S={md.spot:.5f}, F={forward:.5f}, vol={volatility:.2f}%, T={time_to_expiry:.4f}y")
            logger.info(f"  Trade price: {option.trade_price:.6f} ({option.trade_price*100:.4f}%), "
                       f"Direction: {option.direction}, Notional: {option.notional:,.0f}")

            # Price option
            greeks = self.price_option(option, md.spot, forward, volatility)

            logger.info(f"  QuantLib price: {greeks.price:.6f} ({greeks.price*100:.4f}%), "
                       f"delta={greeks.delta:.4f}, vega={greeks.vega:.6f}")

            # Calculate forward delta
            delta = self.calculate_forward_delta(option, md.spot, forward, volatility)

            # Calculate gamma 1%
            gamma_1pct = self.calculate_gamma_1pct(option, md.spot, forward, volatility, vol_surface)

            # Calculate vega (pass market_data for EUR conversion)
            vega_usd, vega_eur = self.calculate_vega(
                option, md.spot, forward, volatility, option.notional, market_data
            )

            logger.info(f"  Forward delta: {delta:.4f}, gamma_1pct: {gamma_1pct:.6f}, "
                       f"vega_usd: {vega_usd:,.0f}, vega_eur: {vega_eur:,.0f}")

            # Update option with calculated values
            option.current_price = greeks.price
            option.delta = delta
            option.delta_notional = delta * option.signed_notional
            option.gamma_1pct = gamma_1pct * option.signed_notional
            option.vega_usd = vega_usd
            option.vega_eur = vega_eur

            # Calculate P&L (pass market_data for EUR conversion)
            option.pnl = self.calculate_pnl(option, greeks.price, market_data)

            logger.info(f"  Final values - current_price: {option.current_price:.6f}, "
                       f"delta_notional: {option.delta_notional:,.0f}, "
                       f"gamma_1pct: {option.gamma_1pct:,.0f}, vega_eur: {option.vega_eur:,.0f}, "
                       f"pnl: {option.pnl:,.0f} EUR")

        return options
