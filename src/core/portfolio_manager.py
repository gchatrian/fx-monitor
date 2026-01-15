"""Portfolio manager for FX options."""

import pandas as pd
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import os

from ..utils.config_loader import ConfigLoader
from ..utils.date_utils import DateUtils
from .pricing_engine import FXOption, FXOptionPricer
from .volatility_surface import VolatilitySurface
from .market_data_provider import FXMarketData

logger = logging.getLogger(__name__)


@dataclass
class AggregatedPosition:
    """Aggregated position for a cross/expiry combination."""
    cross: str
    expiry: date
    total_delta: float = 0.0
    total_delta_notional: float = 0.0
    total_gamma_1pct: float = 0.0
    total_vega_usd: float = 0.0
    total_vega_eur: float = 0.0
    total_pnl: float = 0.0
    options: List[FXOption] = field(default_factory=list)

    @property
    def expiry_str(self) -> str:
        return self.expiry.strftime('%Y-%m-%d')


@dataclass
class CrossSummary:
    """Summary for a single cross across all expiries."""
    cross: str
    total_delta: float = 0.0
    total_delta_notional: float = 0.0
    total_gamma_1pct: float = 0.0
    total_vega_usd: float = 0.0
    total_vega_eur: float = 0.0
    total_pnl: float = 0.0
    positions_by_expiry: Dict[str, AggregatedPosition] = field(default_factory=dict)


class PortfolioManager:
    """
    Manages the FX options portfolio.

    Handles loading from CSV, pricing, and aggregation.
    """

    def __init__(self):
        self.config = ConfigLoader()
        self.pricer = FXOptionPricer()
        self.options: List[FXOption] = []
        self.positions_by_cross_expiry: Dict[Tuple[str, str], AggregatedPosition] = {}
        self.summaries_by_cross: Dict[str, CrossSummary] = {}

    def load_from_csv(self, filepath: str = None) -> List[FXOption]:
        """
        Load options portfolio from CSV file.

        Expected columns:
        Date, Cross, Expiry, Long/Short, Call/Put, Strike, Notional, Price

        Args:
            filepath: Path to CSV file (uses config default if None)

        Returns:
            List of FXOption objects
        """
        if filepath is None:
            filepath = self.config.options_file

        if not os.path.exists(filepath):
            logger.warning(f"Options file not found: {filepath}")
            return []

        try:
            df = pd.read_csv(filepath)

            # Normalize column names
            df.columns = [col.strip().lower().replace('/', '_') for col in df.columns]

            self.options = []

            for _, row in df.iterrows():
                try:
                    option = FXOption(
                        trade_date=self._parse_date(row.get('date', row.get('trade_date', ''))),
                        cross=str(row.get('cross', '')).upper().strip(),
                        expiry=self._parse_date(row.get('expiry', '')),
                        direction=str(row.get('long_short', row.get('direction', 'Long'))).strip(),
                        option_type=str(row.get('call_put', row.get('type', row.get('option_type', 'Call')))).strip(),
                        strike=float(row.get('strike', 0)),
                        notional=float(row.get('notional', 0)),
                        trade_price=float(row.get('price', row.get('trade_price', row.get('premium', 0))))
                    )
                    self.options.append(option)

                except Exception as e:
                    logger.error(f"Error parsing option row: {row}, error: {e}")
                    continue

            logger.info(f"Loaded {len(self.options)} options from {filepath}")
            return self.options

        except Exception as e:
            logger.error(f"Error loading options CSV: {e}")
            return []

    def _parse_date(self, date_str) -> date:
        """Parse date string to date object."""
        if isinstance(date_str, (date, datetime)):
            return date_str if isinstance(date_str, date) else date_str.date()

        date_str = str(date_str).strip()

        for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%Y%m%d', '%m/%d/%Y']:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        raise ValueError(f"Cannot parse date: {date_str}")

    def price_portfolio(self, market_data: Dict[str, FXMarketData],
                       vol_surfaces: Dict[str, VolatilitySurface]) -> None:
        """
        Price all options in the portfolio.

        Args:
            market_data: Dict of {cross: FXMarketData}
            vol_surfaces: Dict of {cross: VolatilitySurface}
        """
        self.options = self.pricer.price_portfolio(self.options, market_data, vol_surfaces)
        self._aggregate_positions()

    def _aggregate_positions(self) -> None:
        """Aggregate positions by cross and expiry."""
        self.positions_by_cross_expiry.clear()
        self.summaries_by_cross.clear()

        for option in self.options:
            key = (option.cross, option.expiry_str)

            if key not in self.positions_by_cross_expiry:
                self.positions_by_cross_expiry[key] = AggregatedPosition(
                    cross=option.cross,
                    expiry=option.expiry
                )

            pos = self.positions_by_cross_expiry[key]
            pos.options.append(option)
            pos.total_delta += option.delta * (1 if option.is_long else -1)
            pos.total_delta_notional += option.delta_notional
            pos.total_gamma_1pct += option.gamma_1pct
            pos.total_vega_usd += option.vega_usd
            pos.total_vega_eur += option.vega_eur
            pos.total_pnl += option.pnl

        # Build cross summaries
        for (cross, expiry_str), pos in self.positions_by_cross_expiry.items():
            if cross not in self.summaries_by_cross:
                self.summaries_by_cross[cross] = CrossSummary(cross=cross)

            summary = self.summaries_by_cross[cross]
            summary.positions_by_expiry[expiry_str] = pos
            summary.total_delta += pos.total_delta
            summary.total_delta_notional += pos.total_delta_notional
            summary.total_gamma_1pct += pos.total_gamma_1pct
            summary.total_vega_usd += pos.total_vega_usd
            summary.total_vega_eur += pos.total_vega_eur
            summary.total_pnl += pos.total_pnl

    def get_position(self, cross: str, expiry: str) -> Optional[AggregatedPosition]:
        """Get aggregated position for a specific cross/expiry."""
        return self.positions_by_cross_expiry.get((cross, expiry))

    def get_cross_summary(self, cross: str) -> Optional[CrossSummary]:
        """Get summary for a cross across all expiries."""
        return self.summaries_by_cross.get(cross)

    def get_all_crosses(self) -> List[str]:
        """Get list of all crosses in portfolio."""
        return list(self.summaries_by_cross.keys())

    def get_expiries_for_cross(self, cross: str) -> List[str]:
        """Get list of expiries for a specific cross."""
        summary = self.summaries_by_cross.get(cross)
        if summary:
            return sorted(summary.positions_by_expiry.keys())
        return []

    def get_options_for_position(self, cross: str, expiry: str) -> List[FXOption]:
        """Get individual options for a position."""
        pos = self.positions_by_cross_expiry.get((cross, expiry))
        if pos:
            return pos.options
        return []

    def add_option(self, option: FXOption) -> None:
        """Add a new option to the portfolio."""
        self.options.append(option)
        self._aggregate_positions()

    def save_to_csv(self, filepath: str = None) -> bool:
        """
        Save portfolio to CSV file.

        Args:
            filepath: Path to CSV file (uses config default if None)

        Returns:
            True if successful
        """
        if filepath is None:
            filepath = self.config.options_file

        try:
            data = []
            for option in self.options:
                data.append({
                    'Date': option.trade_date.strftime('%Y-%m-%d'),
                    'Cross': option.cross,
                    'Expiry': option.expiry.strftime('%Y-%m-%d'),
                    'Long/Short': option.direction,
                    'Call/Put': option.option_type,
                    'Strike': option.strike,
                    'Notional': option.notional,
                    'Price': option.trade_price
                })

            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)

            logger.info(f"Saved {len(self.options)} options to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving options CSV: {e}")
            return False

    def get_portfolio_summary(self) -> Dict:
        """
        Get overall portfolio summary.

        Returns:
            Dict with total metrics across all positions
        """
        total_pnl = sum(s.total_pnl for s in self.summaries_by_cross.values())
        total_vega_usd = sum(s.total_vega_usd for s in self.summaries_by_cross.values())
        total_vega_eur = sum(s.total_vega_eur for s in self.summaries_by_cross.values())

        return {
            'total_pnl': total_pnl,
            'total_vega_usd': total_vega_usd,
            'total_vega_eur': total_vega_eur,
            'num_options': len(self.options),
            'num_crosses': len(self.summaries_by_cross),
            'crosses': list(self.summaries_by_cross.keys())
        }

    def get_delta_ladder(self, cross: str) -> List[Dict]:
        """
        Get delta ladder for a cross (delta by expiry).

        Returns:
            List of dicts with expiry and delta info
        """
        summary = self.summaries_by_cross.get(cross)
        if not summary:
            return []

        ladder = []
        for expiry_str in sorted(summary.positions_by_expiry.keys()):
            pos = summary.positions_by_expiry[expiry_str]
            ladder.append({
                'expiry': expiry_str,
                'delta': pos.total_delta,
                'delta_notional': pos.total_delta_notional,
                'gamma_1pct': pos.total_gamma_1pct
            })

        return ladder
