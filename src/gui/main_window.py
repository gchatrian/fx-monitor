"""Main window for FX Options Portfolio Monitor."""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QPushButton, QLabel, QStatusBar, QHeaderView,
    QSplitter, QFrame, QMessageBox, QProgressDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from datetime import datetime, date
from typing import Dict, Optional
import logging

from .styles import (
    MAIN_STYLESHEET, COLORS, get_pnl_color, format_number,
    format_delta, format_pnl
)
from .detail_window import DetailWindow
from .add_forward_dialog import AddForwardDialog
from ..core.bloomberg_client import BloombergClient, get_bloomberg_client, FXMarketData
from ..core.volatility_surface import VolatilitySurface
from ..core.portfolio_manager import PortfolioManager
from ..core.forward_manager import ForwardManager
from ..utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window."""

    refresh_completed = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.config = ConfigLoader()
        self.bloomberg = get_bloomberg_client()
        self.portfolio_manager = PortfolioManager()
        self.forward_manager = ForwardManager()

        self.market_data: Dict[str, FXMarketData] = {}
        self.vol_surfaces: Dict[str, VolatilitySurface] = {}

        self.detail_windows: Dict[str, DetailWindow] = {}

        self._setup_ui()
        self._setup_connections()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        # Start timer with configured interval (convert to milliseconds)
        self.refresh_timer.start(self.config.refresh_interval * 1000)

        # Initial data load
        QTimer.singleShot(100, self.initial_load)

    def _setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("FX Options Portfolio Monitor")
        self.setMinimumSize(1400, 800)
        self.setStyleSheet(MAIN_STYLESHEET)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Header
        header_layout = QHBoxLayout()

        title_label = QLabel("FX Options Portfolio Monitor")
        title_label.setObjectName("header")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Connection status
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet(f"color: {COLORS['negative']};")
        header_layout.addWidget(self.status_label)

        # Buttons
        self.refresh_btn = QPushButton("Refresh Data")
        self.refresh_btn.setObjectName("refreshButton")
        header_layout.addWidget(self.refresh_btn)

        self.add_forward_btn = QPushButton("Add Forward")
        self.add_forward_btn.setObjectName("addButton")
        header_layout.addWidget(self.add_forward_btn)

        main_layout.addLayout(header_layout)

        # Summary section
        summary_frame = QFrame()
        summary_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        summary_layout = QHBoxLayout(summary_frame)

        # Portfolio summary cards
        self.total_pnl_label = self._create_summary_card("Total P&L", "0.00")
        self.total_vega_label = self._create_summary_card("Total Vega (USD)", "0.00")
        self.options_count_label = self._create_summary_card("Options", "0")
        self.forwards_count_label = self._create_summary_card("Forwards", "0")
        self.last_update_label = self._create_summary_card("Last Update", "--:--:--")

        summary_layout.addWidget(self.total_pnl_label)
        summary_layout.addWidget(self.total_vega_label)
        summary_layout.addWidget(self.options_count_label)
        summary_layout.addWidget(self.forwards_count_label)
        summary_layout.addWidget(self.last_update_label)

        main_layout.addWidget(summary_frame)

        # Main table
        table_label = QLabel("Positions by Cross / Expiry")
        table_label.setObjectName("sectionHeader")
        main_layout.addWidget(table_label)

        self.positions_table = QTableWidget()
        self._setup_positions_table()
        main_layout.addWidget(self.positions_table)

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

    def _create_summary_card(self, title: str, value: str) -> QFrame:
        """Create a summary card widget."""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface_light']};
                border-radius: 6px;
                padding: 10px;
                min-width: 150px;
            }}
        """)
        layout = QVBoxLayout(frame)
        layout.setSpacing(5)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_lbl)

        value_lbl = QLabel(value)
        value_lbl.setStyleSheet(f"color: {COLORS['text']}; font-size: 16px; font-weight: bold;")
        value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_lbl.setObjectName("value")
        layout.addWidget(value_lbl)

        return frame

    def _setup_positions_table(self):
        """Configure the positions table."""
        columns = [
            "Cross", "Expiry",
            "Options P&L", "Forward P&L", "Total P&L",
            "Options Delta", "Forward Delta", "Total Delta",
            "Gamma 1%", "Vega USD", "Vega EUR"
        ]

        self.positions_table.setColumnCount(len(columns))
        self.positions_table.setHorizontalHeaderLabels(columns)
        self.positions_table.setAlternatingRowColors(True)
        self.positions_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.positions_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        # Header configuration
        header = self.positions_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setMinimumSectionSize(100)

        # Enable sorting
        self.positions_table.setSortingEnabled(True)

        # Double-click to show details
        self.positions_table.cellDoubleClicked.connect(self._on_row_double_click)

    def _setup_connections(self):
        """Set up signal connections."""
        self.refresh_btn.clicked.connect(self.refresh_data)
        self.add_forward_btn.clicked.connect(self._show_add_forward_dialog)
        self.refresh_completed.connect(self._on_refresh_completed)

    def initial_load(self):
        """Load initial data."""
        self.statusBar.showMessage("Loading portfolio data...")

        # Load portfolios from CSV
        self.portfolio_manager.load_from_csv()
        self.forward_manager.load_from_csv()

        # Try to connect to Bloomberg
        self._connect_bloomberg()

        # Refresh market data
        self.refresh_data()

    def _connect_bloomberg(self):
        """Connect to Bloomberg API."""
        try:
            if self.bloomberg.connect():
                self.status_label.setText("Connected")
                self.status_label.setStyleSheet(f"color: {COLORS['positive']};")
                self.statusBar.showMessage("Connected to Bloomberg")
            else:
                self.status_label.setText("Disconnected")
                self.status_label.setStyleSheet(f"color: {COLORS['negative']};")
                self.statusBar.showMessage("Bloomberg connection failed - using cached data")
        except Exception as e:
            logger.error(f"Bloomberg connection error: {e}")
            self.status_label.setText("Error")
            self.status_label.setStyleSheet(f"color: {COLORS['warning']};")

    def refresh_data(self):
        """Refresh all market data and recalculate positions."""
        self.statusBar.showMessage("Refreshing market data...")
        self.refresh_btn.setEnabled(False)

        try:
            # Get all crosses from portfolio
            option_crosses = set(self.portfolio_manager.get_all_crosses())
            forward_crosses = set(self.forward_manager.get_all_crosses())
            all_crosses = option_crosses.union(forward_crosses)

            if not all_crosses:
                all_crosses = set(self.config.fx_crosses)

            # Fetch market data
            for cross in all_crosses:
                try:
                    md = self.bloomberg.get_all_market_data(cross, force_refresh=True)
                    self.market_data[cross] = md

                    # Build volatility surface
                    if md.vol_surface:
                        vol_surface = VolatilitySurface(cross)
                        vol_surface.build_from_market_data(
                            md.spot,
                            md.forward_rates,
                            md.vol_surface
                        )
                        self.vol_surfaces[cross] = vol_surface

                except Exception as e:
                    logger.error(f"Error fetching data for {cross}: {e}")

            # Price options
            self.portfolio_manager.price_portfolio(self.market_data, self.vol_surfaces)

            # Calculate forward positions
            self.forward_manager.calculate_positions(self.bloomberg)

            # Update UI
            self._update_positions_table()
            self._update_summary()

            self.refresh_completed.emit()

        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            self.statusBar.showMessage(f"Refresh error: {e}")

        finally:
            self.refresh_btn.setEnabled(True)

    def _on_refresh_completed(self):
        """Handle refresh completion."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.statusBar.showMessage(f"Data refreshed at {timestamp}")

        # Update last update time
        value_label = self.last_update_label.findChild(QLabel, "value")
        if value_label:
            value_label.setText(timestamp)

    def _update_positions_table(self):
        """Update the positions table with current data."""
        self.positions_table.setSortingEnabled(False)
        self.positions_table.setRowCount(0)

        # Combine option and forward positions
        all_positions = {}

        # Add option positions
        for cross in self.portfolio_manager.get_all_crosses():
            for expiry in self.portfolio_manager.get_expiries_for_cross(cross):
                key = (cross, expiry)
                opt_pos = self.portfolio_manager.get_position(cross, expiry)

                if key not in all_positions:
                    all_positions[key] = {
                        'cross': cross,
                        'expiry': expiry,
                        'opt_pnl': 0, 'fwd_pnl': 0,
                        'opt_delta': 0, 'fwd_delta': 0,
                        'gamma': 0, 'vega_usd': 0, 'vega_eur': 0
                    }

                if opt_pos:
                    all_positions[key]['opt_pnl'] = opt_pos.total_pnl
                    all_positions[key]['opt_delta'] = opt_pos.total_delta_notional
                    all_positions[key]['gamma'] = opt_pos.total_gamma_1pct
                    all_positions[key]['vega_usd'] = opt_pos.total_vega_usd
                    all_positions[key]['vega_eur'] = opt_pos.total_vega_eur

        # Add forward positions (match by expiry/value_date)
        for cross in self.forward_manager.get_all_crosses():
            for value_date in self.forward_manager.get_dates_for_cross(cross):
                key = (cross, value_date)

                if key not in all_positions:
                    all_positions[key] = {
                        'cross': cross,
                        'expiry': value_date,
                        'opt_pnl': 0, 'fwd_pnl': 0,
                        'opt_delta': 0, 'fwd_delta': 0,
                        'gamma': 0, 'vega_usd': 0, 'vega_eur': 0
                    }

                fwd_pos = self.forward_manager.get_position(cross, value_date)
                if fwd_pos:
                    all_positions[key]['fwd_pnl'] = fwd_pos.total_pnl
                    all_positions[key]['fwd_delta'] = fwd_pos.total_delta

        # Populate table
        for (cross, expiry), data in sorted(all_positions.items()):
            row = self.positions_table.rowCount()
            self.positions_table.insertRow(row)

            total_pnl = data['opt_pnl'] + data['fwd_pnl']
            total_delta = data['opt_delta'] + data['fwd_delta']

            items = [
                (cross, None),
                (expiry, None),
                (format_pnl(data['opt_pnl']), get_pnl_color(data['opt_pnl'])),
                (format_pnl(data['fwd_pnl']), get_pnl_color(data['fwd_pnl'])),
                (format_pnl(total_pnl), get_pnl_color(total_pnl)),
                (format_number(data['opt_delta'], 0), None),
                (format_number(data['fwd_delta'], 0), None),
                (format_number(total_delta, 0), get_pnl_color(total_delta)),
                (format_number(data['gamma'], 0), None),
                (format_number(data['vega_usd'], 0), None),
                (format_number(data['vega_eur'], 0), None),
            ]

            for col, (text, color) in enumerate(items):
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(QColor(color))
                self.positions_table.setItem(row, col, item)

        self.positions_table.setSortingEnabled(True)

    def _update_summary(self):
        """Update summary cards."""
        # Portfolio summary
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        blotter_summary = self.forward_manager.get_blotter_summary()

        total_pnl = portfolio_summary['total_pnl'] + blotter_summary['total_pnl']

        # Update cards
        self._update_card_value(self.total_pnl_label, format_pnl(total_pnl),
                               get_pnl_color(total_pnl))
        self._update_card_value(self.total_vega_label,
                               format_number(portfolio_summary['total_vega_usd'], 0))
        self._update_card_value(self.options_count_label,
                               str(portfolio_summary['num_options']))
        self._update_card_value(self.forwards_count_label,
                               str(blotter_summary['num_forwards']))

    def _update_card_value(self, card: QFrame, value: str, color: str = None):
        """Update a summary card's value."""
        value_label = card.findChild(QLabel, "value")
        if value_label:
            value_label.setText(value)
            if color:
                value_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")

    def _on_row_double_click(self, row: int, col: int):
        """Handle double-click on a table row."""
        cross_item = self.positions_table.item(row, 0)
        expiry_item = self.positions_table.item(row, 1)

        if cross_item and expiry_item:
            cross = cross_item.text()
            expiry = expiry_item.text()
            self._show_detail_window(cross, expiry)

    def _show_detail_window(self, cross: str, expiry: str):
        """Show detail window for a position."""
        key = f"{cross}_{expiry}"

        if key in self.detail_windows:
            # Bring existing window to front
            self.detail_windows[key].raise_()
            self.detail_windows[key].activateWindow()
        else:
            # Create new detail window
            opt_pos = self.portfolio_manager.get_position(cross, expiry)
            fwd_pos = self.forward_manager.get_position(cross, expiry)
            md = self.market_data.get(cross)

            window = DetailWindow(cross, expiry, opt_pos, fwd_pos, md, self)
            window.closed.connect(lambda: self.detail_windows.pop(key, None))
            window.forward_added.connect(self._on_forward_added)
            self.detail_windows[key] = window
            window.show()

    def _show_add_forward_dialog(self):
        """Show dialog to add a new forward."""
        dialog = AddForwardDialog(self.config.fx_crosses, self)
        if dialog.exec():
            forward = dialog.get_forward()
            if forward:
                self._add_forward(forward)

    def _on_forward_added(self, forward):
        """Handle forward added from detail window."""
        self._add_forward(forward)

    def _add_forward(self, forward):
        """Add a forward to the blotter and save."""
        self.forward_manager.add_forward(forward)
        self.forward_manager.save_to_csv()

        # Recalculate and refresh
        self.forward_manager.calculate_positions(self.bloomberg)
        self._update_positions_table()
        self._update_summary()

        self.statusBar.showMessage(f"Forward added: {forward.direction} {format_number(forward.notional, 0)} {forward.cross}")

    def closeEvent(self, event):
        """Handle window close."""
        # Close all detail windows
        for window in list(self.detail_windows.values()):
            window.close()

        # Stop refresh timer
        self.refresh_timer.stop()

        # Disconnect Bloomberg
        self.bloomberg.disconnect()

        event.accept()
