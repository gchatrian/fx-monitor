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
        # Use absolute delta convention: ATM at 0.50, wings at lower deltas
        # This creates a symmetric smile around ATM
        # Deltas: 0.10 (10d wing), 0.25 (25d), 0.50 (ATM)
        # Put vols are on left side, call vols on right side
        # We store both sides separately for interpolation
        self.put_deltas = np.array([0.10, 0.25, 0.50])  # 10P, 25P, ATM
        self.put_vols = np.array([self.vol_10p, self.vol_25p, self.atm_vol])

        self.call_deltas = np.array([0.10, 0.25, 0.50])  # 10C, 25C, ATM
        self.call_vols = np.array([self.vol_10c, self.vol_25c, self.atm_vol])

        # Create separate interpolators for put and call sides
        self._put_interp = interpolate.CubicSpline(
            self.put_deltas,
            self.put_vols,
            bc_type='natural'
        )
        self._call_interp = interpolate.CubicSpline(
            self.call_deltas,
            self.call_vols,
            bc_type='natural'
        )

        logger.debug(f"VolSmile {self.tenor}: Built smile - "
                    f"10P={self.vol_10p:.2f}%, 25P={self.vol_25p:.2f}%, ATM={self.atm_vol:.2f}%, "
                    f"25C={self.vol_25c:.2f}%, 10C={self.vol_10c:.2f}%")

    def get_vol_for_delta(self, delta: float) -> float:
        """
        Get interpolated volatility for a given delta.

        The smile uses absolute delta convention:
        - Put side: delta from -0.50 (ATM) to -0.10 (10d OTM put)
        - Call side: delta from +0.50 (ATM) to +0.10 (10d OTM call)

        We use the absolute value of delta and select the appropriate
        interpolator (put or call) based on the sign.

        Args:
            delta: Option delta (-1 to 1, negative for puts)

        Returns:
            Interpolated volatility in %
        """
        original_delta = delta

        # Convert to absolute delta for interpolation
        abs_delta = abs(delta)

        # Clamp to valid range [0.10, 0.50]
        if abs_delta > 0.50:
            abs_delta = 0.50  # ITM options use ATM vol
        elif abs_delta < 0.10:
            abs_delta = 0.10  # Deep OTM options use wing vol

        # Select interpolator based on put/call
        if delta < 0:
            # Put side
            result = float(self._put_interp(abs_delta))
        else:
            # Call side (including delta=0, which defaults to ATM)
            result = float(self._call_interp(abs_delta))

        if abs(abs(original_delta) - abs_delta) > 0.01:
            logger.debug(f"VolSmile {self.tenor}: get_vol_for_delta({original_delta:.4f}) clamped to {abs_delta:.4f} = {result:.2f}%")
        else:
            logger.debug(f"VolSmile {self.tenor}: get_vol_for_delta({original_delta:.4f}) = {result:.2f}%")

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

    def build_strike_grid(self) -> Dict[str, float]:
        """
        Calcola gli strike per ogni delta pillar.

        Returns:
            Dict mapping pillar names to strike prices:
            {'10P': strike_10p, '25P': strike_25p, 'ATM': strike_atm,
             '25C': strike_25c, '10C': strike_10c}
        """
        if self.forward <= 0 or self.time_to_expiry <= 0:
            # Return forward as fallback for all strikes
            return {
                '10P': self.forward,
                '25P': self.forward,
                'ATM': self.forward,
                '25C': self.forward,
                '10C': self.forward
            }

        strikes = {}

        # Per ogni delta, calcola lo strike usando la vol corrispondente
        # 10 delta put (delta = -0.10)
        strikes['10P'] = self.delta_to_strike(-0.10, self.vol_10p)
        strikes['25P'] = self.delta_to_strike(-0.25, self.vol_25p)
        strikes['ATM'] = self.delta_to_strike(0.50, self.atm_vol)
        strikes['25C'] = self.delta_to_strike(0.25, self.vol_25c)
        strikes['10C'] = self.delta_to_strike(0.10, self.vol_10c)

        logger.debug(f"VolSmile {self.tenor}: Strike grid - "
                    f"10P={strikes['10P']:.5f}, 25P={strikes['25P']:.5f}, "
                    f"ATM={strikes['ATM']:.5f}, 25C={strikes['25C']:.5f}, "
                    f"10C={strikes['10C']:.5f}")

        return strikes


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
        # Strike/vol grid for 2D interpolation
        self._grid_times: List[float] = []
        self._grid_strikes: List[List[float]] = []
        self._grid_vols: List[List[float]] = []
        self._strike_interps: List = []

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
        self._build_strike_vol_grid()
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

    def _build_strike_vol_grid(self) -> None:
        """
        Costruisce griglia (strike, time) -> vol per interpolazione 2D.

        Per ogni tenor, calcola gli strike corrispondenti ai 5 delta pillars
        (10P, 25P, ATM, 25C, 10C) e memorizza la griglia per interpolazione.
        """
        if not self.smiles:
            return

        self._grid_times = []       # Lista di tempi
        self._grid_strikes = []     # Lista di strike per ogni tempo (ordinati)
        self._grid_vols = []        # Lista di vol per ogni tempo
        self._strike_interps = []   # Interpolatori cubici strike->vol per ogni tenor

        for tenor in self._sorted_tenors:
            smile = self.smiles[tenor]
            t = smile.time_to_expiry

            # Calcola strike per i 5 delta pillars
            strike_grid = smile.build_strike_grid()

            # Ordina per strike crescente (10P < 25P < ATM < 25C < 10C)
            strikes = [
                strike_grid['10P'],
                strike_grid['25P'],
                strike_grid['ATM'],
                strike_grid['25C'],
                strike_grid['10C']
            ]
            vols = [
                smile.vol_10p,
                smile.vol_25p,
                smile.atm_vol,
                smile.vol_25c,
                smile.vol_10c
            ]

            self._grid_times.append(t)
            self._grid_strikes.append(strikes)
            self._grid_vols.append(vols)

            # Crea interpolatore cubico strike->vol per questo tenor
            strike_interp = interpolate.CubicSpline(
                strikes, vols, bc_type='natural'
            )
            self._strike_interps.append(strike_interp)

            logger.debug(f"VolSurface {self.cross} {tenor}: Strike grid built - "
                        f"K=[{strikes[0]:.5f}, {strikes[-1]:.5f}], "
                        f"Vol=[{vols[0]:.2f}%, {vols[-1]:.2f}%]")

    def _find_surrounding_tenors(self, time_to_expiry: float) -> Tuple[int, int]:
        """
        Trova gli indici dei due tenor circostanti per un dato time_to_expiry.

        Returns:
            Tuple (idx1, idx2) degli indici dei tenor circostanti.
            Se time_to_expiry è fuori range, ritorna lo stesso indice per entrambi.
        """
        if time_to_expiry <= self._grid_times[0]:
            return (0, 0)

        if time_to_expiry >= self._grid_times[-1]:
            last_idx = len(self._grid_times) - 1
            return (last_idx, last_idx)

        for i in range(len(self._grid_times) - 1):
            if self._grid_times[i] <= time_to_expiry <= self._grid_times[i + 1]:
                return (i, i + 1)

        return (0, 0)

    def _interpolate_smile_by_strike(self, tenor_idx: int, strike: float) -> float:
        """
        Interpola vol per uno strike dato, usando l'interpolatore cubico del tenor.

        Args:
            tenor_idx: Indice del tenor nella griglia
            strike: Strike price

        Returns:
            Volatilità interpolata in %
        """
        strikes = self._grid_strikes[tenor_idx]
        vols = self._grid_vols[tenor_idx]

        # Extrapola flat ai bordi
        if strike <= strikes[0]:
            return vols[0]  # Wing put (10P vol)
        if strike >= strikes[-1]:
            return vols[-1]  # Wing call (10C vol)

        # Usa interpolatore cubico
        return float(self._strike_interps[tenor_idx](strike))

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
        Get volatility for a specific strike and expiry using strike-based interpolation.

        Interpola direttamente sulla griglia strike/vol costruita dai delta pillars,
        evitando la conversione circolare strike->delta->vol.

        Processo:
        1. Trova i due tenor circostanti (t1, t2)
        2. Per ogni tenor, interpola vol vs strike usando spline cubica
        3. Interpola linearmente in volatilità tra i due tenor

        Args:
            time_to_expiry: Time to expiry in years
            strike: Strike price
            forward: Forward rate for this expiry (not used, kept for API compatibility)
            is_call: True for call option (not used, sticky-strike convention)

        Returns:
            Interpolated volatility in %
        """
        if not self._grid_times or strike <= 0:
            return self.get_vol(time_to_expiry, 0.5)

        # Handle edge cases
        if time_to_expiry <= 0:
            time_to_expiry = 1/365  # Minimum 1 day

        # Trova i tenor circostanti
        t1_idx, t2_idx = self._find_surrounding_tenors(time_to_expiry)

        # Interpola vol vs strike per ciascun tenor
        vol1 = self._interpolate_smile_by_strike(t1_idx, strike)

        # Se siamo esattamente su un tenor o fuori range, ritorna direttamente
        if t1_idx == t2_idx:
            logger.debug(f"VolSurface.get_vol_for_strike: T={time_to_expiry:.4f}y, K={strike:.5f}, "
                        f"single tenor idx={t1_idx}, vol={vol1:.2f}%")
            return vol1

        vol2 = self._interpolate_smile_by_strike(t2_idx, strike)

        # Interpola linearmente in volatilità tra i due tenor
        t1 = self._grid_times[t1_idx]
        t2 = self._grid_times[t2_idx]

        w = (time_to_expiry - t1) / (t2 - t1)
        result = vol1 + w * (vol2 - vol1)

        logger.debug(f"VolSurface.get_vol_for_strike: T={time_to_expiry:.4f}y, K={strike:.5f}, "
                    f"t1={t1:.4f}y (vol={vol1:.2f}%), t2={t2:.4f}y (vol={vol2:.2f}%), "
                    f"interpolated={result:.2f}%")

        return result

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
