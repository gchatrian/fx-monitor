"""Date utilities for FX options calculations."""

import QuantLib as ql
from datetime import datetime, date, timedelta
from typing import Union
import re


class DateUtils:
    """Utility class for date operations and tenor parsing."""

    # Calendar for FX (typically TARGET for EUR pairs)
    _calendars = {
        'EUR': ql.TARGET(),
        'USD': ql.UnitedStates(ql.UnitedStates.FederalReserve),
        'GBP': ql.UnitedKingdom(),
        'JPY': ql.Japan(),
        'CHF': ql.Switzerland(),
        'CAD': ql.Canada(),
        'AUD': ql.Australia(),
        'NZD': ql.NewZealand(),
    }

    @staticmethod
    def to_ql_date(dt: Union[datetime, date, str]) -> ql.Date:
        """Convert Python date to QuantLib Date."""
        if isinstance(dt, str):
            # Try different date formats
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%Y%m%d']:
                try:
                    dt = datetime.strptime(dt, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Cannot parse date: {dt}")

        if isinstance(dt, datetime):
            dt = dt.date()

        return ql.Date(dt.day, dt.month, dt.year)

    @staticmethod
    def from_ql_date(ql_date: ql.Date) -> date:
        """Convert QuantLib Date to Python date."""
        return date(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())

    @staticmethod
    def parse_tenor(tenor: str) -> tuple:
        """
        Parse a tenor string into (count, period_type).

        Examples:
            '1W' -> (1, 'W')
            '3M' -> (3, 'M')
            '1Y' -> (1, 'Y')
            'ON' -> overnight
            'TN' -> tom-next
            'SN' -> spot-next
        """
        tenor = tenor.upper().strip()

        if tenor == 'ON':
            return (1, 'D')
        elif tenor == 'TN':
            return (2, 'D')
        elif tenor == 'SN':
            return (3, 'D')

        match = re.match(r'^(\d+)([DWMY])$', tenor)
        if not match:
            raise ValueError(f"Invalid tenor format: {tenor}")

        count = int(match.group(1))
        period = match.group(2)

        return (count, period)

    @staticmethod
    def tenor_to_ql_period(tenor: str) -> ql.Period:
        """Convert tenor string to QuantLib Period."""
        count, period_type = DateUtils.parse_tenor(tenor)

        period_map = {
            'D': ql.Days,
            'W': ql.Weeks,
            'M': ql.Months,
            'Y': ql.Years,
        }

        return ql.Period(count, period_map[period_type])

    @staticmethod
    def add_tenor(start_date: Union[datetime, date, ql.Date], tenor: str,
                  calendar: ql.Calendar = None) -> ql.Date:
        """Add a tenor to a date using business day convention."""
        if not isinstance(start_date, ql.Date):
            start_date = DateUtils.to_ql_date(start_date)

        if calendar is None:
            calendar = ql.TARGET()

        period = DateUtils.tenor_to_ql_period(tenor)
        return calendar.advance(start_date, period, ql.ModifiedFollowing)

    @staticmethod
    def year_fraction(start_date: Union[datetime, date, ql.Date],
                     end_date: Union[datetime, date, ql.Date],
                     day_count: ql.DayCounter = None) -> float:
        """Calculate year fraction between two dates."""
        if not isinstance(start_date, ql.Date):
            start_date = DateUtils.to_ql_date(start_date)
        if not isinstance(end_date, ql.Date):
            end_date = DateUtils.to_ql_date(end_date)

        if day_count is None:
            day_count = ql.Actual365Fixed()

        return day_count.yearFraction(start_date, end_date)

    @staticmethod
    def tenor_to_years(tenor: str) -> float:
        """Convert tenor string to approximate year fraction."""
        count, period_type = DateUtils.parse_tenor(tenor)

        multipliers = {
            'D': 1/365,
            'W': 7/365,
            'M': 1/12,
            'Y': 1.0,
        }

        return count * multipliers.get(period_type, 0)

    @staticmethod
    def get_fx_calendar(ccy_pair: str) -> ql.Calendar:
        """Get the appropriate calendar for an FX pair."""
        ccy1 = ccy_pair[:3]
        ccy2 = ccy_pair[3:]

        cal1 = DateUtils._calendars.get(ccy1, ql.TARGET())
        cal2 = DateUtils._calendars.get(ccy2, ql.TARGET())

        return ql.JointCalendar(cal1, cal2)

    @staticmethod
    def get_spot_date(trade_date: Union[datetime, date, ql.Date],
                     ccy_pair: str) -> ql.Date:
        """Calculate spot date for an FX pair (T+2 for most pairs, T+1 for some)."""
        if not isinstance(trade_date, ql.Date):
            trade_date = DateUtils.to_ql_date(trade_date)

        calendar = DateUtils.get_fx_calendar(ccy_pair)

        # Most pairs are T+2, USD/CAD is T+1
        spot_lag = 1 if 'CAD' in ccy_pair and 'USD' in ccy_pair else 2

        return calendar.advance(trade_date, spot_lag, ql.Days)

    @staticmethod
    def days_between(start_date: Union[datetime, date, ql.Date],
                    end_date: Union[datetime, date, ql.Date]) -> int:
        """Calculate number of calendar days between two dates."""
        if not isinstance(start_date, ql.Date):
            start_date = DateUtils.to_ql_date(start_date)
        if not isinstance(end_date, ql.Date):
            end_date = DateUtils.to_ql_date(end_date)

        return end_date - start_date
