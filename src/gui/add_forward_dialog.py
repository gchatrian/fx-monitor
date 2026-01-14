"""Dialog for adding a new FX forward position."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox,
    QDoubleSpinBox, QDateEdit, QPushButton, QLabel, QFrame,
    QMessageBox, QLineEdit
)
from PyQt6.QtCore import Qt, QDate
from PyQt6.QtGui import QDoubleValidator
from datetime import date, datetime
from typing import List, Optional

from .styles import MAIN_STYLESHEET, COLORS
from ..core.forward_manager import FXForward


class AddForwardDialog(QDialog):
    """Dialog for adding a new FX forward."""

    def __init__(self, crosses: List[str], parent=None,
                 default_cross: str = None, default_expiry: str = None):
        super().__init__(parent)

        self.crosses = crosses
        self.default_cross = default_cross
        self.default_expiry = default_expiry
        self.forward: Optional[FXForward] = None

        self._setup_ui()

    def _setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Add FX Forward")
        self.setMinimumWidth(400)
        self.setStyleSheet(MAIN_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(25, 25, 25, 25)

        # Header
        header = QLabel("Add New Forward Position")
        header.setStyleSheet(f"""
            color: {COLORS['accent']};
            font-size: 16px;
            font-weight: bold;
            padding-bottom: 10px;
            border-bottom: 1px solid {COLORS['border']};
        """)
        layout.addWidget(header)

        # Form
        form_frame = QFrame()
        form_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-radius: 8px;
                padding: 15px;
            }}
        """)
        form_layout = QFormLayout(form_frame)
        form_layout.setSpacing(15)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Cross selection
        self.cross_combo = QComboBox()
        self.cross_combo.addItems(self.crosses)
        if self.default_cross and self.default_cross in self.crosses:
            self.cross_combo.setCurrentText(self.default_cross)
        form_layout.addRow("Currency Pair:", self.cross_combo)

        # Direction
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Buy", "Sell"])
        form_layout.addRow("Direction:", self.direction_combo)

        # Notional
        self.notional_edit = QLineEdit()
        self.notional_edit.setPlaceholderText("e.g., 1000000")
        self.notional_edit.setValidator(QDoubleValidator(0, 1e15, 2))
        form_layout.addRow("Notional:", self.notional_edit)

        # Value date
        self.value_date_edit = QDateEdit()
        self.value_date_edit.setCalendarPopup(True)
        self.value_date_edit.setDisplayFormat("yyyy-MM-dd")

        if self.default_expiry:
            try:
                expiry_date = datetime.strptime(self.default_expiry, '%Y-%m-%d').date()
                self.value_date_edit.setDate(QDate(expiry_date.year, expiry_date.month, expiry_date.day))
            except:
                self.value_date_edit.setDate(QDate.currentDate().addMonths(1))
        else:
            self.value_date_edit.setDate(QDate.currentDate().addMonths(1))

        form_layout.addRow("Value Date:", self.value_date_edit)

        # Rate (optional)
        self.rate_edit = QLineEdit()
        self.rate_edit.setPlaceholderText("e.g., 1.0850 (optional)")
        self.rate_edit.setValidator(QDoubleValidator(0, 1000, 6))
        form_layout.addRow("Trade Rate:", self.rate_edit)

        layout.addWidget(form_frame)

        # Example label
        example_label = QLabel(
            "Example: Buy 1,000,000 EURUSD value date 14/04/2026 @ 1.1701"
        )
        example_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        example_label.setWordWrap(True)
        layout.addWidget(example_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        button_layout.addStretch()

        add_btn = QPushButton("Add Forward")
        add_btn.setObjectName("addButton")
        add_btn.clicked.connect(self._add_forward)
        button_layout.addWidget(add_btn)

        layout.addLayout(button_layout)

    def _add_forward(self):
        """Validate and create the forward."""
        # Validate notional
        notional_text = self.notional_edit.text().strip()
        if not notional_text:
            QMessageBox.warning(self, "Validation Error", "Please enter a notional amount.")
            return

        try:
            notional = float(notional_text.replace(',', ''))
            if notional <= 0:
                raise ValueError("Notional must be positive")
        except ValueError as e:
            QMessageBox.warning(self, "Validation Error", f"Invalid notional: {e}")
            return

        # Get rate (optional)
        rate_text = self.rate_edit.text().strip()
        rate = 0.0
        if rate_text:
            try:
                rate = float(rate_text)
            except ValueError:
                QMessageBox.warning(self, "Validation Error", "Invalid rate format.")
                return

        # Get value date
        value_date_qt = self.value_date_edit.date()
        value_date = date(value_date_qt.year(), value_date_qt.month(), value_date_qt.day())

        if value_date <= date.today():
            QMessageBox.warning(self, "Validation Error", "Value date must be in the future.")
            return

        # Create forward
        self.forward = FXForward(
            trade_date=date.today(),
            cross=self.cross_combo.currentText(),
            direction=self.direction_combo.currentText(),
            notional=notional,
            value_date=value_date,
            rate=rate
        )

        self.accept()

    def get_forward(self) -> Optional[FXForward]:
        """Get the created forward."""
        return self.forward
