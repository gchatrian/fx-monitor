"""Forward blotter manager for FX forwards."""

import pandas as pd
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import os

from ..utils.config_loader import ConfigLoader
from ..utils.date_utils import DateUtils
from .bloomberg_client import BloombergClient, get_bloomberg_client

logger = logging.getLogger(__name__)


@dataclass
class FXForward:
    """Represents an FX forward position."""
    trade_date: date
    cross: str
    direction: str  # 'Buy' or 'Sell'
    notional: float
    value_date: date
    rate: float = 0.0  # Trade rate (optional)

    # Calculated fields
    current_rate: float = 0.0
    delta_equivalent: float = 0.0
    pnl: float = 0.0

    def __post_init__(self):
        """Normalize input values."""
        self.direction = self.direction.capitalize()

        # Convert string dates if necessary
        if isinstance(self.trade_date, str):
            self.trade_date = self._parse_date(self.trade_date)
        if isinstance(self.value_date, str):
            self.value_date = self._parse_date(self.value_date)

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

    @property
    def is_buy(self) -> bool:
        return self.direction.lower() == 'buy'

    @property
    def signed_notional(self) -> float:
        """Notional with sign based on direction."""
        return self.notional if self.is_buy else -self.notional

    @property
    def value_date_str(self) -> str:
        """Return value date as string for grouping."""
        return self.value_date.strftime('%Y-%m-%d')


@dataclass
class AggregatedForwardPosition:
    """Aggregated forward position for a cross/value_date combination."""
    cross: str
    value_date: date
    net_notional: float = 0.0
    total_delta: float = 0.0
    total_pnl: float = 0.0
    forwards: List[FXForward] = field(default_factory=list)

    @property
    def value_date_str(self) -> str:
        return self.value_date.strftime('%Y-%m-%d')


@dataclass
class CrossForwardSummary:
    """Summary for a single cross across all value dates."""
    cross: str
    net_notional: float = 0.0
    total_delta: float = 0.0
    total_pnl: float = 0.0
    positions_by_date: Dict[str, AggregatedForwardPosition] = field(default_factory=dict)


class ForwardManager:
    """
    Manages the FX forward blotter.

    Handles loading from CSV, P&L calculation, and aggregation.
    """

    def __init__(self):
        self.config = ConfigLoader()
        self.forwards: List[FXForward] = []
        self.positions_by_cross_date: Dict[Tuple[str, str], AggregatedForwardPosition] = {}
        self.summaries_by_cross: Dict[str, CrossForwardSummary] = {}

    def load_from_csv(self, filepath: str = None) -> List[FXForward]:
        """
        Load forward blotter from CSV file.

        Expected columns:
        Date, Cross, Buy/Sell, Notional, Value Date, Rate (optional)

        Args:
            filepath: Path to CSV file (uses config default if None)

        Returns:
            List of FXForward objects
        """
        if filepath is None:
            filepath = self.config.forwards_file

        if not os.path.exists(filepath):
            logger.warning(f"Forwards file not found: {filepath}")
            return []

        try:
            df = pd.read_csv(filepath)

            # Normalize column names
            df.columns = [col.strip().lower().replace(' ', '_').replace('/', '_') for col in df.columns]

            self.forwards = []

            for _, row in df.iterrows():
                try:
                    forward = FXForward(
                        trade_date=row.get('date', row.get('trade_date', date.today())),
                        cross=str(row.get('cross', '')).upper().strip(),
                        direction=str(row.get('buy_sell', row.get('direction', row.get('buy_or_sell', 'Buy')))).strip(),
                        notional=float(row.get('notional', 0)),
                        value_date=row.get('value_date', row.get('maturity', '')),
                        rate=float(row.get('rate', row.get('trade_rate', 0))) if pd.notna(row.get('rate', row.get('trade_rate', None))) else 0.0
                    )
                    self.forwards.append(forward)

                except Exception as e:
                    logger.error(f"Error parsing forward row: {row}, error: {e}")
                    continue

            logger.info(f"Loaded {len(self.forwards)} forwards from {filepath}")
            return self.forwards

        except Exception as e:
            logger.error(f"Error loading forwards CSV: {e}")
            return []

    def calculate_positions(self, bloomberg_client: BloombergClient = None) -> None:
        """
        Calculate P&L and aggregate positions.

        Args:
            bloomberg_client: Optional Bloomberg client for market data
        """
        if bloomberg_client is None:
            bloomberg_client = get_bloomberg_client()

        today = date.today()

        for forward in self.forwards:
            # Get current forward rate for value date
            if bloomberg_client.is_connected():
                forward.current_rate = bloomberg_client.get_forward_rate_for_date(
                    forward.cross, forward.value_date
                ) or 0.0
            else:
                forward.current_rate = forward.rate  # Use trade rate as fallback

            # Calculate P&L
            if forward.rate > 0 and forward.current_rate > 0:
                # P&L = notional * (current_rate - trade_rate) for buy
                # P&L = notional * (trade_rate - current_rate) for sell
                rate_diff = forward.current_rate - forward.rate
                pip_factor = 10000 if 'JPY' not in forward.cross else 100

                if forward.is_buy:
                    forward.pnl = forward.notional * rate_diff
                else:
                    forward.pnl = forward.notional * (-rate_diff)

            # Delta equivalent is simply the signed notional
            forward.delta_equivalent = forward.signed_notional

        self._aggregate_positions()

    def _aggregate_positions(self) -> None:
        """Aggregate positions by cross and value date."""
        self.positions_by_cross_date.clear()
        self.summaries_by_cross.clear()

        for forward in self.forwards:
            key = (forward.cross, forward.value_date_str)

            if key not in self.positions_by_cross_date:
                self.positions_by_cross_date[key] = AggregatedForwardPosition(
                    cross=forward.cross,
                    value_date=forward.value_date
                )

            pos = self.positions_by_cross_date[key]
            pos.forwards.append(forward)
            pos.net_notional += forward.signed_notional
            pos.total_delta += forward.delta_equivalent
            pos.total_pnl += forward.pnl

        # Build cross summaries
        for (cross, date_str), pos in self.positions_by_cross_date.items():
            if cross not in self.summaries_by_cross:
                self.summaries_by_cross[cross] = CrossForwardSummary(cross=cross)

            summary = self.summaries_by_cross[cross]
            summary.positions_by_date[date_str] = pos
            summary.net_notional += pos.net_notional
            summary.total_delta += pos.total_delta
            summary.total_pnl += pos.total_pnl

    def get_position(self, cross: str, value_date: str) -> Optional[AggregatedForwardPosition]:
        """Get aggregated position for a specific cross/value_date."""
        return self.positions_by_cross_date.get((cross, value_date))

    def get_cross_summary(self, cross: str) -> Optional[CrossForwardSummary]:
        """Get summary for a cross across all value dates."""
        return self.summaries_by_cross.get(cross)

    def get_all_crosses(self) -> List[str]:
        """Get list of all crosses in blotter."""
        return list(self.summaries_by_cross.keys())

    def get_dates_for_cross(self, cross: str) -> List[str]:
        """Get list of value dates for a specific cross."""
        summary = self.summaries_by_cross.get(cross)
        if summary:
            return sorted(summary.positions_by_date.keys())
        return []

    def get_forwards_for_position(self, cross: str, value_date: str) -> List[FXForward]:
        """Get individual forwards for a position."""
        pos = self.positions_by_cross_date.get((cross, value_date))
        if pos:
            return pos.forwards
        return []

    def add_forward(self, forward: FXForward) -> None:
        """Add a new forward to the blotter."""
        self.forwards.append(forward)
        self._aggregate_positions()

    def save_to_csv(self, filepath: str = None) -> bool:
        """
        Save blotter to CSV file.

        Args:
            filepath: Path to CSV file (uses config default if None)

        Returns:
            True if successful
        """
        if filepath is None:
            filepath = self.config.forwards_file

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            data = []
            for forward in self.forwards:
                data.append({
                    'Date': forward.trade_date.strftime('%Y-%m-%d'),
                    'Cross': forward.cross,
                    'Buy/Sell': forward.direction,
                    'Notional': forward.notional,
                    'Value Date': forward.value_date.strftime('%Y-%m-%d'),
                    'Rate': forward.rate if forward.rate > 0 else ''
                })

            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)

            logger.info(f"Saved {len(self.forwards)} forwards to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving forwards CSV: {e}")
            return False

    def get_delta_for_expiry(self, cross: str, expiry_date: str) -> float:
        """
        Get total forward delta for a cross matching an option expiry.

        Matches forwards where value_date equals expiry_date.
        """
        pos = self.positions_by_cross_date.get((cross, expiry_date))
        if pos:
            return pos.total_delta
        return 0.0

    def get_pnl_for_expiry(self, cross: str, expiry_date: str) -> float:
        """
        Get total forward P&L for a cross matching an option expiry.
        """
        pos = self.positions_by_cross_date.get((cross, expiry_date))
        if pos:
            return pos.total_pnl
        return 0.0

    def get_blotter_summary(self) -> Dict:
        """
        Get overall blotter summary.

        Returns:
            Dict with total metrics across all positions
        """
        total_pnl = sum(s.total_pnl for s in self.summaries_by_cross.values())
        total_notional = sum(abs(s.net_notional) for s in self.summaries_by_cross.values())

        return {
            'total_pnl': total_pnl,
            'total_notional': total_notional,
            'num_forwards': len(self.forwards),
            'num_crosses': len(self.summaries_by_cross),
            'crosses': list(self.summaries_by_cross.keys())
        }

    def calculate_positions_with_provider(self, market_provider) -> None:
        """
        Calculate P&L and aggregate positions using MarketDataProvider.

        Args:
            market_provider: MarketDataProvider instance (supports both Bloomberg and mock)
        """
        for forward in self.forwards:
            # Get current forward rate for value date
            forward.current_rate = market_provider.get_forward_rate_for_date(
                forward.cross, forward.value_date
            )

            # Calculate P&L
            if forward.rate > 0 and forward.current_rate > 0:
                rate_diff = forward.current_rate - forward.rate

                if forward.is_buy:
                    forward.pnl = forward.notional * rate_diff
                else:
                    forward.pnl = forward.notional * (-rate_diff)

            # Delta equivalent is simply the signed notional
            forward.delta_equivalent = forward.signed_notional

        self._aggregate_positions()
