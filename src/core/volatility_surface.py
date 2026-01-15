"""Volatility surface construction and interpolation for FX options."""

import numpy as np
from scipy import interpolate
from scipy.optimize import brentq
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging
import math

from ..utils.date_utils import DateUtils

logger = logging.getLogger(__name__)


@dataclass
class VolSmilePoint:
    """A single point on the volatility smile."""
    delta: float  # Delta (0-1 for calls, -1 to 0 for puts)
    vol: float    # Implied volatility in %
    strike: float = 0.0  # Calculated strike


class VolatilitySmile:
    """Volatility smile for a single expiry."""

    def __init__(self, tenor: str, atm_vol: float, rr_25: float, rr_10: float,
                 bf_25: float, bf_10: float, spot: float = 0.0,
                 forward: float = 0.0, time_to_expiry: float = 0.0):
        """
        Initialize volatility smile from market quotes.

        Args:
            tenor: Tenor string (e.g., '1M', '3M')
            atm_vol: ATM volatility in %
            rr_25: 25 delta risk reversal (25d call vol - 25d put vol)
            rr_10: 10 delta risk reversal
            bf_25: 25 delta butterfly ((25d call + 25d put)/2 - ATM)
            bf_10: 10 delta butterfly
            spot: Spot rate
            forward: Forward rate for this tenor
            time_to_expiry: Time to expiry in years
        """
        self.tenor = tenor
        self.atm_vol = atm_vol
        self.rr_25 = rr_25
        self.rr_10 = rr_10
        self.bf_25 = bf_25
        self.bf_10 = bf_10
        self.spot = spot
        self.forward = forward
        self.time_to_expiry = time_to_expiry

        # Calculate delta-vol points
        self._build_smile()

    def _build_smile(self):
        """Build the volatility smile from market quotes."""
        # Convert RR and BF to individual call/put vols
        # Butterfly: BF = (vol_call + vol_put)/2 - ATM
        # Risk Reversal: RR = vol_call - vol_put

        # For 25 delta:
        # bf_25 = (vol_25c + vol_25p)/2 - atm => vol_25c + vol_25p = 2*(bf_25 + atm)
        # rr_25 = vol_25c - vol_25p
        # => vol_25c = bf_25 + atm + rr_25/2
        # => vol_25p = bf_25 + atm - rr_25/2

        self.vol_25c = self.atm_vol + self.bf_25 + self.rr_25 / 2
        self.vol_25p = self.atm_vol + self.bf_25 - self.rr_25 / 2

        self.vol_10c = self.atm_vol + self.bf_10 + self.rr_10 / 2
        self.vol_10p = self.atm_vol + self.bf_10 - self.rr_10 / 2

        # Build delta-vol array for interpolation
        # Deltas: 10p, 25p, ATM (50d), 25c, 10c
        # Using premium-adjusted deltas convention
        self.deltas = np.array([-0.10, -0.25, 0.50, 0.25, 0.10])
        self.vols = np.array([
            self.vol_10p,
            self.vol_25p,
            self.atm_vol,
            self.vol_25c,
            self.vol_10c
        ])

        # Create cubic spline interpolator
        # Sort by delta for interpolation
        sort_idx = np.argsort(self.deltas)
        self._interp = interpolate.CubicSpline(
            self.deltas[sort_idx],
            self.vols[sort_idx],
            bc_type='natural'
        )

        logger.debug(f"VolSmile {self.tenor}: Built smile - "
                    f"10P={self.vol_10p:.2f}%, 25P={self.vol_25p:.2f}%, ATM={self.atm_vol:.2f}%, "
                    f"25C={self.vol_25c:.2f}%, 10C={self.vol_10c:.2f}%")

    def get_vol_for_delta(self, delta: float) -> float:
        """
        Get interpolated volatility for a given delta.

        Args:
            delta: Option delta (-1 to 1, negative for puts)

        Returns:
            Interpolated volatility in %
        """
        # Clamp delta to reasonable range
        delta = max(-0.95, min(0.95, delta))

        # Handle near-zero delta (deep OTM)
        if abs(delta) < 0.05:
            # Extrapolate using the wing
            if delta >= 0:
                return float(self._interp(0.10))
            else:
                return float(self._interp(-0.10))

        result = float(self._interp(delta))
        logger.debug(f"VolSmile {self.tenor}: get_vol_for_delta({delta:.4f}) = {result:.2f}% "
                    f"[ATM={self.atm_vol:.2f}%, 25c={self.vol_25c:.2f}%, 25p={self.vol_25p:.2f}%]")
        return result

    def get_vol_for_strike(self, strike: float) -> float:
        """
        Get volatility for a given strike using delta interpolation.

        This requires solving for the delta that corresponds to the strike.
        """
        if self.forward <= 0 or self.time_to_expiry <= 0:
            return self.atm_vol

        # Use Black-Scholes to find delta for this strike
        delta = self._strike_to_delta(strike)
        return self.get_vol_for_delta(delta)

    def _strike_to_delta(self, strike: float) -> float:
        """Convert strike to delta using BS formula."""
        if self.forward <= 0 or self.time_to_expiry <= 0:
            return 0.5

        # Initial guess using ATM vol
        vol = self.atm_vol / 100  # Convert to decimal
        sqrt_t = math.sqrt(self.time_to_expiry)

        log_moneyness = math.log(self.forward / strike)
        d1 = (log_moneyness + 0.5 * vol * vol * self.time_to_expiry) / (vol * sqrt_t)

        # Forward delta
        from scipy.stats import norm
        delta = norm.cdf(d1)

        # Convert to put delta if strike > forward (OTM put)
        if strike > self.forward:
            delta = delta - 1

        return delta

    def delta_to_strike(self, delta: float, vol: float = None) -> float:
        """
        Convert delta to strike using BS formula.

        Args:
            delta: Option delta
            vol: Volatility to use (if None, interpolate from surface)

        Returns:
            Strike price
        """
        if self.forward <= 0 or self.time_to_expiry <= 0:
            return self.forward

        if vol is None:
            vol = self.get_vol_for_delta(delta)

        vol_decimal = vol / 100
        sqrt_t = math.sqrt(self.time_to_expiry)

        from scipy.stats import norm

        # For calls (positive delta), for puts (negative delta)
        if delta > 0:
            d1 = norm.ppf(delta)
        else:
            d1 = norm.ppf(1 + delta)  # Put delta is negative

        # K = F * exp(-d1 * vol * sqrt(T) + 0.5 * vol^2 * T)
        strike = self.forward * math.exp(-d1 * vol_decimal * sqrt_t + 0.5 * vol_decimal**2 * self.time_to_expiry)

        return strike


class VolatilitySurface:
    """
    Complete FX volatility surface with tenor and strike interpolation.
    """

    def __init__(self, cross: str):
        """
        Initialize volatility surface.

        Args:
            cross: FX pair (e.g., 'EURUSD')
        """
        self.cross = cross
        self.spot = 0.0
        self.smiles: Dict[str, VolatilitySmile] = {}
        self.tenor_times: Dict[str, float] = {}
        self._time_interpolator = None

    def add_smile(self, tenor: str, smile: VolatilitySmile) -> None:
        """Add a volatility smile for a specific tenor."""
        self.smiles[tenor] = smile
        self.tenor_times[tenor] = smile.time_to_expiry

    def build_from_market_data(self, spot: float, forward_rates: Dict[str, float],
                               vol_data: Dict[str, Dict[str, float]]) -> None:
        """
        Build the volatility surface from Bloomberg market data.

        Args:
            spot: Spot rate
            forward_rates: Dict of {tenor: forward_rate}
            vol_data: Dict of {tenor: {pillar: vol}}
                      where pillar is ATM, 25RR, 10RR, 25BF, 10BF
        """
        self.spot = spot
        self.smiles.clear()
        self.tenor_times.clear()

        for tenor, pillars in vol_data.items():
            # Get volatility pillars with defaults
            atm = pillars.get('ATM', 10.0)
            rr_25 = pillars.get('25RR', 0.0)
            rr_10 = pillars.get('10RR', 0.0)
            bf_25 = pillars.get('25BF', 0.0)
            bf_10 = pillars.get('10BF', 0.0)

            # Get forward rate for this tenor
            forward = forward_rates.get(tenor, spot)

            # Calculate time to expiry
            time_to_expiry = DateUtils.tenor_to_years(tenor)

            smile = VolatilitySmile(
                tenor=tenor,
                atm_vol=atm,
                rr_25=rr_25,
                rr_10=rr_10,
                bf_25=bf_25,
                bf_10=bf_10,
                spot=spot,
                forward=forward,
                time_to_expiry=time_to_expiry
            )

            self.smiles[tenor] = smile
            self.tenor_times[tenor] = time_to_expiry
            logger.info(f"VolSurface {self.cross}: Added {tenor} smile - ATM={atm:.2f}%, "
                       f"25RR={rr_25:.2f}, 10RR={rr_10:.2f}, 25BF={bf_25:.2f}, 10BF={bf_10:.2f}")
            logger.info(f"  -> Smile points: 10P={smile.vol_10p:.2f}%, 25P={smile.vol_25p:.2f}%, "
                       f"ATM={smile.atm_vol:.2f}%, 25C={smile.vol_25c:.2f}%, 10C={smile.vol_10c:.2f}%")

        self._build_time_interpolator()
        logger.info(f"VolSurface {self.cross}: Built with tenors {self._sorted_tenors} "
                   f"times {[f'{t:.4f}y' for t in self._sorted_times]}")

    def _build_time_interpolator(self) -> None:
        """Build interpolator for time dimension."""
        if not self.smiles:
            return

        # Sort tenors by time
        sorted_tenors = sorted(self.tenor_times.items(), key=lambda x: x[1])
        self._sorted_tenors = [t[0] for t in sorted_tenors]
        self._sorted_times = np.array([t[1] for t in sorted_tenors])

    def get_vol(self, time_to_expiry: float, delta: float = 0.5) -> float:
        """
        Get interpolated volatility for given expiry and delta.

        Args:
            time_to_expiry: Time to expiry in years
            delta: Option delta (-1 to 1)

        Returns:
            Interpolated volatility in %
        """
        if not self.smiles:
            return 10.0  # Default vol

        # Handle edge cases
        if time_to_expiry <= 0:
            time_to_expiry = 1/365  # Minimum 1 day

        # Find surrounding tenors
        if time_to_expiry <= self._sorted_times[0]:
            # Before first tenor - use first smile
            return self.smiles[self._sorted_tenors[0]].get_vol_for_delta(delta)

        if time_to_expiry >= self._sorted_times[-1]:
            # After last tenor - use flat extrapolation (last vol)
            # This is more conservative than linear extrapolation
            return self.smiles[self._sorted_tenors[-1]].get_vol_for_delta(delta)

        # Linear interpolation in volatility between surrounding tenors
        for i in range(len(self._sorted_times) - 1):
            t1 = self._sorted_times[i]
            t2 = self._sorted_times[i + 1]

            if t1 <= time_to_expiry <= t2:
                tenor1 = self._sorted_tenors[i]
                tenor2 = self._sorted_tenors[i + 1]
                v1 = self.smiles[tenor1].get_vol_for_delta(delta)
                v2 = self.smiles[tenor2].get_vol_for_delta(delta)

                # Linear interpolation in volatility
                w = (time_to_expiry - t1) / (t2 - t1)
                result = v1 + w * (v2 - v1)

                logger.debug(f"VolSurface.get_vol: T={time_to_expiry:.4f}y between {tenor1}({t1:.4f}y) "
                            f"and {tenor2}({t2:.4f}y), delta={delta:.2f}, v1={v1:.2f}%, v2={v2:.2f}%, "
                            f"interpolated={result:.2f}%")
                return result

        return self.smiles[self._sorted_tenors[0]].get_vol_for_delta(delta)

    def get_vol_for_strike(self, time_to_expiry: float, strike: float,
                          forward: float, is_call: bool = True) -> float:
        """
        Get volatility for a specific strike and expiry.

        Args:
            time_to_expiry: Time to expiry in years
            strike: Strike price
            forward: Forward rate for this expiry
            is_call: True for call option, False for put option

        Returns:
            Interpolated volatility in %
        """
        if forward <= 0 or strike <= 0:
            return self.get_vol(time_to_expiry, 0.5)

        # First get ATM vol to estimate delta
        atm_vol = self.get_vol(time_to_expiry, 0.5)

        # Calculate approximate delta using this vol
        vol_decimal = atm_vol / 100
        sqrt_t = math.sqrt(max(time_to_expiry, 1/365))

        from scipy.stats import norm

        log_moneyness = math.log(forward / strike)
        d1 = (log_moneyness + 0.5 * vol_decimal * vol_decimal * time_to_expiry) / (vol_decimal * sqrt_t)

        # Calculate delta based on option type
        # For calls: delta = N(d1), positive (0 to 1)
        # For puts: delta = N(d1) - 1, negative (-1 to 0)
        if is_call:
            delta = norm.cdf(d1)  # Call delta: positive
        else:
            delta = norm.cdf(d1) - 1  # Put delta: negative

        # Now get vol for this delta
        final_vol = self.get_vol(time_to_expiry, delta)

        logger.debug(f"VolSurface.get_vol_for_strike: T={time_to_expiry:.4f}y, K={strike:.5f}, "
                    f"F={forward:.5f}, is_call={is_call}, atm_vol={atm_vol:.2f}%, delta={delta:.4f}, final_vol={final_vol:.2f}%")

        return final_vol

    def get_forward_for_expiry(self, time_to_expiry: float) -> float:
        """
        Get interpolated forward rate for a specific expiry.

        Args:
            time_to_expiry: Time to expiry in years

        Returns:
            Interpolated forward rate
        """
        if not self.smiles:
            return self.spot

        # Find surrounding tenors and interpolate forwards
        if time_to_expiry <= self._sorted_times[0]:
            return self.smiles[self._sorted_tenors[0]].forward

        if time_to_expiry >= self._sorted_times[-1]:
            # Extrapolate using forward points relationship
            return self.smiles[self._sorted_tenors[-1]].forward

        for i in range(len(self._sorted_times) - 1):
            t1 = self._sorted_times[i]
            t2 = self._sorted_times[i + 1]

            if t1 <= time_to_expiry <= t2:
                f1 = self.smiles[self._sorted_tenors[i]].forward
                f2 = self.smiles[self._sorted_tenors[i + 1]].forward

                # Linear interpolation in forward points
                w = (time_to_expiry - t1) / (t2 - t1)
                return f1 + w * (f2 - f1)

        return self.spot
