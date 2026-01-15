"""Detail window for FX position breakdown."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QFrame, QHeaderView, QTabWidget, QGroupBox,
    QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from typing import Optional
from datetime import date

from .styles import (
    MAIN_STYLESHEET, COLORS, get_pnl_color, format_number,
    format_delta, format_pnl
)
from .add_forward_dialog import AddForwardDialog
from ..core.portfolio_manager import AggregatedPosition
from ..core.forward_manager import AggregatedForwardPosition, FXForward
from ..core.market_data_provider import FXMarketData
from ..core.volatility_surface import VolatilitySurface


class DetailWindow(QWidget):
    """Detail window showing position breakdown for a cross/expiry."""

    closed = pyqtSignal()
    forward_added = pyqtSignal(object)  # FXForward

    def __init__(self, cross: str, expiry: str,
                 option_position: Optional[AggregatedPosition],
                 forward_position: Optional[AggregatedForwardPosition],
                 market_data: Optional[FXMarketData],
                 vol_surface: Optional[VolatilitySurface] = None,
                 parent=None):
        super().__init__(parent)

        self.cross = cross
        self.expiry = expiry
        self.option_position = option_position
        self.forward_position = forward_position
        self.market_data = market_data
        self.vol_surface = vol_surface

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle(f"Position Detail - {self.cross} {self.expiry}")
        self.setMinimumSize(900, 500)
        self.resize(1000, 550)
        self.setStyleSheet(MAIN_STYLESHEET)
        self.setWindowFlags(Qt.WindowType.Window)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header_layout = QHBoxLayout()

        title_label = QLabel(f"{self.cross} - Expiry: {self.expiry}")
        title_label.setObjectName("header")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Add forward button
        add_fwd_btn = QPushButton("Add Forward")
        add_fwd_btn.setObjectName("addButton")
        add_fwd_btn.clicked.connect(self._show_add_forward)
        header_layout.addWidget(add_fwd_btn)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        header_layout.addWidget(close_btn)

        layout.addLayout(header_layout)

        # Market data section
        if self.market_data:
            market_frame = self._create_market_section()
            layout.addWidget(market_frame)

        # Tab widget for details
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {COLORS['border']};
                background-color: {COLORS['surface']};
            }}
        """)

        # Options tab
        if self.option_position and self.option_position.options:
            options_widget = self._create_options_tab()
            tab_widget.addTab(options_widget, "Options Details")

        # Forwards tab
        if self.forward_position and self.forward_position.forwards:
            forwards_widget = self._create_forwards_tab()
            tab_widget.addTab(forwards_widget, "Forwards Details")

        layout.addWidget(tab_widget)

    def _create_market_section(self) -> QFrame:
        """Create market data display section."""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-radius: 8px;
                padding: 10px;
            }}
        """)

        layout = QHBoxLayout(frame)

        # Spot
        spot_box = self._create_info_box("Spot", f"{self.market_data.spot:.5f}")
        layout.addWidget(spot_box)

        # Forward rate for expiry (interpolated from vol surface)
        forward_rate = self._get_forward_for_expiry()
        if forward_rate:
            fwd_box = self._create_info_box("Forward", f"{forward_rate:.5f}")
            layout.addWidget(fwd_box)

        # ATM Vol (if available)
        if self.market_data.vol_surface:
            first_tenor = list(self.market_data.vol_surface.keys())[0] if self.market_data.vol_surface else None
            if first_tenor and 'ATM' in self.market_data.vol_surface.get(first_tenor, {}):
                atm_vol = self.market_data.vol_surface[first_tenor]['ATM']
                vol_box = self._create_info_box("ATM Vol", f"{atm_vol:.2f}%")
                layout.addWidget(vol_box)

        # Last update
        update_box = self._create_info_box("Updated",
                                           self.market_data.timestamp.strftime("%H:%M:%S"))
        layout.addWidget(update_box)

        layout.addStretch()

        return frame

    def _create_info_box(self, label: str, value: str) -> QFrame:
        """Create an information display box."""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface_light']};
                border-radius: 4px;
                padding: 8px;
                min-width: 100px;
            }}
        """)

        layout = QVBoxLayout(frame)
        layout.setSpacing(2)
        layout.setContentsMargins(8, 8, 8, 8)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

        val = QLabel(value)
        val.setStyleSheet(f"color: {COLORS['accent']}; font-size: 14px; font-weight: bold;")
        val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(val)

        return frame

    def _create_summary_section(self) -> QFrame:
        """Create position summary section."""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-radius: 8px;
                padding: 15px;
            }}
        """)

        layout = QGridLayout(frame)
        layout.setSpacing(20)

        # Calculate totals
        opt_pnl = self.option_position.total_pnl if self.option_position else 0
        opt_delta = self.option_position.total_delta_notional if self.option_position else 0
        opt_gamma = self.option_position.total_gamma_1pct if self.option_position else 0
        opt_vega_usd = self.option_position.total_vega_usd if self.option_position else 0
        opt_vega_eur = self.option_position.total_vega_eur if self.option_position else 0

        fwd_pnl = self.forward_position.total_pnl if self.forward_position else 0
        fwd_delta = self.forward_position.total_delta if self.forward_position else 0

        total_pnl = opt_pnl + fwd_pnl
        total_delta = opt_delta + fwd_delta

        # Headers
        headers = ["", "Options", "Forwards", "Total"]
        for col, header in enumerate(headers):
            lbl = QLabel(header)
            lbl.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl, 0, col)

        # P&L row
        self._add_summary_row(layout, 1, "P&L",
                             format_pnl(opt_pnl), get_pnl_color(opt_pnl),
                             format_pnl(fwd_pnl), get_pnl_color(fwd_pnl),
                             format_pnl(total_pnl), get_pnl_color(total_pnl))

        # Delta row
        self._add_summary_row(layout, 2, "Delta",
                             format_number(opt_delta, 0), None,
                             format_number(fwd_delta, 0), None,
                             format_number(total_delta, 0), get_pnl_color(total_delta))

        # Options-only metrics
        layout.addWidget(QLabel(""), 3, 0)  # Spacer

        gamma_lbl = QLabel("Gamma 1%")
        gamma_lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(gamma_lbl, 4, 0)
        gamma_val = QLabel(format_number(opt_gamma, 0))
        gamma_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(gamma_val, 4, 1)

        vega_usd_lbl = QLabel("Vega USD")
        vega_usd_lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(vega_usd_lbl, 5, 0)
        vega_usd_val = QLabel(format_number(opt_vega_usd, 0))
        vega_usd_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(vega_usd_val, 5, 1)

        vega_eur_lbl = QLabel("Vega EUR")
        vega_eur_lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(vega_eur_lbl, 6, 0)
        vega_eur_val = QLabel(format_number(opt_vega_eur, 0))
        vega_eur_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(vega_eur_val, 6, 1)

        return frame

    def _add_summary_row(self, layout: QGridLayout, row: int, label: str,
                        opt_val: str, opt_color: str,
                        fwd_val: str, fwd_color: str,
                        total_val: str, total_color: str):
        """Add a summary row to the grid."""
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(lbl, row, 0)

        opt = QLabel(opt_val)
        opt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if opt_color:
            opt.setStyleSheet(f"color: {opt_color};")
        layout.addWidget(opt, row, 1)

        fwd = QLabel(fwd_val)
        fwd.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if fwd_color:
            fwd.setStyleSheet(f"color: {fwd_color};")
        layout.addWidget(fwd, row, 2)

        total = QLabel(total_val)
        total.setAlignment(Qt.AlignmentFlag.AlignCenter)
        total.setStyleSheet(f"font-weight: bold; color: {total_color or COLORS['text']};")
        layout.addWidget(total, row, 3)

    def _create_options_tab(self) -> QWidget:
        """Create options detail tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        table = QTableWidget()
        columns = [
            "Trade Date", "Type", "Direction", "Strike", "Notional",
            "Price %", "Current %", "Delta", "Gamma 1%", "Vega EUR", "P&L (EUR)"
        ]

        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for option in self.option_position.options:
            row = table.rowCount()
            table.insertRow(row)

            # Format prices as percentage (e.g., 0.0125 -> 1.25%)
            trade_price_pct = f"{option.trade_price * 100:.2f}%"
            current_price_pct = f"{option.current_price * 100:.2f}%"

            items = [
                option.trade_date.strftime('%Y-%m-%d'),
                option.option_type,
                option.direction,
                f"{option.strike:.5f}",
                format_number(option.notional, 0),
                trade_price_pct,
                current_price_pct,
                format_delta(option.delta),
                format_number(option.gamma_1pct, 0),
                format_number(option.vega_eur, 0),
                format_pnl(option.pnl)
            ]

            for col, text in enumerate(items):
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # Color P&L column
                if col == len(items) - 1:
                    item.setForeground(QColor(get_pnl_color(option.pnl)))

                # Color direction
                if col == 2:
                    color = COLORS['positive'] if option.direction == 'Long' else COLORS['negative']
                    item.setForeground(QColor(color))

                table.setItem(row, col, item)

        layout.addWidget(table)
        return widget

    def _create_forwards_tab(self) -> QWidget:
        """Create forwards detail tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        table = QTableWidget()
        columns = [
            "Trade Date", "Direction", "Notional", "Value Date",
            "Trade Rate", "Current Rate", "P&L (EUR)"
        ]

        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for forward in self.forward_position.forwards:
            row = table.rowCount()
            table.insertRow(row)

            items = [
                forward.trade_date.strftime('%Y-%m-%d'),
                forward.direction,
                format_number(forward.notional, 0),
                forward.value_date.strftime('%Y-%m-%d'),
                f"{forward.rate:.5f}" if forward.rate > 0 else "-",
                f"{forward.current_rate:.5f}" if forward.current_rate > 0 else "-",
                format_pnl(forward.pnl)
            ]

            for col, text in enumerate(items):
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # Color P&L column
                if col == len(items) - 1:
                    item.setForeground(QColor(get_pnl_color(forward.pnl)))

                # Color direction
                if col == 1:
                    color = COLORS['positive'] if forward.direction == 'Buy' else COLORS['negative']
                    item.setForeground(QColor(color))

                table.setItem(row, col, item)

        layout.addWidget(table)
        return widget

    def _show_add_forward(self):
        """Show dialog to add a forward for this cross/expiry."""
        dialog = AddForwardDialog([self.cross], self, self.cross, self.expiry)
        if dialog.exec():
            forward = dialog.get_forward()
            if forward:
                self.forward_added.emit(forward)

    def _get_forward_for_expiry(self) -> Optional[float]:
        """Get the forward rate for the expiry date."""
        from datetime import datetime

        if not self.expiry:
            return None

        try:
            # Parse expiry date
            expiry_date = datetime.strptime(self.expiry, '%Y-%m-%d').date()
            today = date.today()

            # Calculate time to expiry in years
            days_to_expiry = (expiry_date - today).days
            if days_to_expiry <= 0:
                return None
            time_to_expiry = days_to_expiry / 365.0

            # Use vol surface if available (has interpolated forward rates)
            if self.vol_surface:
                return self.vol_surface.get_forward_for_expiry(time_to_expiry)

            # Fallback: interpolate from market_data.forward_rates
            if self.market_data and self.market_data.forward_rates:
                from ..utils.date_utils import DateUtils

                # Build tenor -> time mapping
                tenor_times = {}
                for tenor in self.market_data.forward_rates.keys():
                    tenor_times[tenor] = DateUtils.tenor_to_years(tenor)

                # Sort by time
                sorted_tenors = sorted(tenor_times.items(), key=lambda x: x[1])

                if not sorted_tenors:
                    return None

                # Find surrounding tenors
                if time_to_expiry <= sorted_tenors[0][1]:
                    return self.market_data.forward_rates[sorted_tenors[0][0]]

                if time_to_expiry >= sorted_tenors[-1][1]:
                    return self.market_data.forward_rates[sorted_tenors[-1][0]]

                for i in range(len(sorted_tenors) - 1):
                    t1_tenor, t1_time = sorted_tenors[i]
                    t2_tenor, t2_time = sorted_tenors[i + 1]

                    if t1_time <= time_to_expiry <= t2_time:
                        f1 = self.market_data.forward_rates[t1_tenor]
                        f2 = self.market_data.forward_rates[t2_tenor]

                        # Linear interpolation
                        w = (time_to_expiry - t1_time) / (t2_time - t1_time)
                        return f1 + w * (f2 - f1)

            return None

        except Exception:
            return None

    def closeEvent(self, event):
        """Handle window close."""
        self.closed.emit()
        event.accept()
