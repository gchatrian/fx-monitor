"""Professional financial application styles for PyQt6."""

# Dark theme color palette
COLORS = {
    'background': '#1a1a2e',
    'surface': '#16213e',
    'surface_light': '#1f3460',
    'primary': '#0f4c75',
    'primary_light': '#3282b8',
    'accent': '#00d9ff',
    'text': '#e8e8e8',
    'text_secondary': '#a0a0a0',
    'positive': '#00c853',
    'negative': '#ff5252',
    'warning': '#ffc107',
    'border': '#2a2a4a',
    'header': '#0d1b2a',
    'row_alt': '#1e2d4a',
}

MAIN_STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['background']};
}}

QWidget {{
    background-color: {COLORS['background']};
    color: {COLORS['text']};
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    font-size: 12px;
}}

QLabel {{
    color: {COLORS['text']};
    background-color: transparent;
}}

QLabel#header {{
    font-size: 18px;
    font-weight: bold;
    color: {COLORS['accent']};
    padding: 10px;
}}

QLabel#sectionHeader {{
    font-size: 14px;
    font-weight: bold;
    color: {COLORS['primary_light']};
    padding: 5px;
    border-bottom: 1px solid {COLORS['border']};
}}

QPushButton {{
    background-color: {COLORS['primary']};
    color: {COLORS['text']};
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
    min-width: 80px;
}}

QPushButton:hover {{
    background-color: {COLORS['primary_light']};
}}

QPushButton:pressed {{
    background-color: {COLORS['surface_light']};
}}

QPushButton:disabled {{
    background-color: {COLORS['border']};
    color: {COLORS['text_secondary']};
}}

QPushButton#refreshButton {{
    background-color: {COLORS['accent']};
    color: {COLORS['background']};
}}

QPushButton#refreshButton:hover {{
    background-color: #33e5ff;
}}

QPushButton#addButton {{
    background-color: {COLORS['positive']};
    color: {COLORS['background']};
}}

QPushButton#addButton:hover {{
    background-color: #2edf6b;
}}

QTableWidget {{
    background-color: {COLORS['surface']};
    alternate-background-color: {COLORS['row_alt']};
    gridline-color: {COLORS['border']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    selection-background-color: {COLORS['primary']};
}}

QTableWidget::item {{
    padding: 8px;
    border-bottom: 1px solid {COLORS['border']};
}}

QTableWidget::item:selected {{
    background-color: {COLORS['primary']};
}}

QHeaderView::section {{
    background-color: {COLORS['header']};
    color: {COLORS['accent']};
    padding: 10px;
    border: none;
    border-bottom: 2px solid {COLORS['accent']};
    font-weight: bold;
    font-size: 11px;
    text-transform: uppercase;
}}

QScrollBar:vertical {{
    background-color: {COLORS['surface']};
    width: 12px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['primary']};
    border-radius: 6px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['primary_light']};
}}

QScrollBar:horizontal {{
    background-color: {COLORS['surface']};
    height: 12px;
    border-radius: 6px;
}}

QScrollBar::handle:horizontal {{
    background-color: {COLORS['primary']};
    border-radius: 6px;
    min-width: 30px;
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    border: none;
    background: none;
}}

QLineEdit {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 8px;
    color: {COLORS['text']};
}}

QLineEdit:focus {{
    border-color: {COLORS['accent']};
}}

QComboBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 8px;
    color: {COLORS['text']};
    min-width: 100px;
}}

QComboBox:hover {{
    border-color: {COLORS['primary_light']};
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid {COLORS['accent']};
    margin-right: 10px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['primary']};
}}

QDateEdit {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 8px;
    color: {COLORS['text']};
}}

QDateEdit:focus {{
    border-color: {COLORS['accent']};
}}

QDateEdit::drop-down {{
    border: none;
    width: 30px;
}}

QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 8px;
    color: {COLORS['text']};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS['accent']};
}}

QGroupBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 15px;
    font-weight: bold;
}}

QGroupBox::title {{
    color: {COLORS['accent']};
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}}

QTabWidget::pane {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
}}

QTabBar::tab {{
    background-color: {COLORS['header']};
    color: {COLORS['text_secondary']};
    padding: 10px 20px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}}

QTabBar::tab:selected {{
    background-color: {COLORS['primary']};
    color: {COLORS['text']};
}}

QTabBar::tab:hover {{
    background-color: {COLORS['surface_light']};
}}

QStatusBar {{
    background-color: {COLORS['header']};
    color: {COLORS['text_secondary']};
    border-top: 1px solid {COLORS['border']};
}}

QToolTip {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['accent']};
    border-radius: 4px;
    padding: 5px;
}}

QDialog {{
    background-color: {COLORS['background']};
}}

QMessageBox {{
    background-color: {COLORS['background']};
}}

QProgressBar {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    text-align: center;
    color: {COLORS['text']};
}}

QProgressBar::chunk {{
    background-color: {COLORS['accent']};
    border-radius: 3px;
}}
"""


def get_pnl_color(value: float) -> str:
    """Get color for P&L value."""
    if value > 0:
        return COLORS['positive']
    elif value < 0:
        return COLORS['negative']
    return COLORS['text']


def format_number(value: float, decimals: int = 2, prefix: str = '', suffix: str = '') -> str:
    """Format number with thousands separator."""
    if abs(value) >= 1_000_000:
        return f"{prefix}{value/1_000_000:,.{decimals}f}M{suffix}"
    elif abs(value) >= 1_000:
        return f"{prefix}{value/1_000:,.{decimals}f}K{suffix}"
    return f"{prefix}{value:,.{decimals}f}{suffix}"


def format_delta(value: float) -> str:
    """Format delta value."""
    return f"{value:+.4f}"


def format_pnl(value: float) -> str:
    """Format P&L value with sign."""
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:+,.2f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:+,.2f}K"
    return f"{value:+,.2f}"
