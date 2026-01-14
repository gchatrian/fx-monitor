# FX Options Portfolio Monitor

A professional-grade Python application for monitoring and managing FX options portfolios. Built with Bloomberg API for real-time market data and QuantLib for accurate derivatives pricing.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)
![QuantLib](https://img.shields.io/badge/Pricing-QuantLib-orange.svg)
![Bloomberg](https://img.shields.io/badge/Data-Bloomberg%20API-red.svg)

## Features

### Market Data (Bloomberg API)
- Real-time spot rates for all configured FX crosses
- Forward points curve (ON to 1Y tenors)
- Complete volatility surface:
  - ATM volatility
  - 25 Delta and 10 Delta Risk Reversals
  - 25 Delta and 10 Delta Butterflies

### Pricing Engine (QuantLib)
- Garman-Kohlhagen model for FX vanilla options
- Forward Delta (Black-Scholes)
- Vega in USD and EUR terms
- Gamma 1% (delta change for 1% spot move)
- Real-time P&L calculation

### Volatility Surface
- Cubic spline interpolation across delta dimension
- Variance-time interpolation across tenor dimension
- Automatic extrapolation for non-standard expiries
- Strike-to-delta conversion for smile lookup

### Portfolio Management
- CSV-based portfolio storage
- Aggregation by cross and expiry
- Combined options and forwards view
- Delta ladder visualization

### User Interface
- Professional dark theme design
- Real-time data refresh
- Position drill-down with double-click
- Quick forward entry dialog
- Configurable auto-refresh interval

## Installation

### Prerequisites
- Python 3.9 or higher
- Bloomberg Terminal with Desktop API enabled
- Windows, macOS, or Linux

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fx-monitor.git
cd fx-monitor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure Bloomberg Terminal is running and logged in.

5. Run the application:
```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize:

```yaml
bloomberg:
  host: "localhost"
  port: 8194
  timeout: 30000

data:
  directory: "./data/"
  options_file: "options_portfolio.csv"
  forwards_file: "forwards_blotter.csv"

fx_crosses:
  - EURUSD
  - GBPUSD
  - USDJPY
  - EURGBP
  - EURCHF
  - USDCAD
  - AUDUSD
  - AUDNZD

display:
  refresh_interval: 60  # seconds
```

## Data Files

### Options Portfolio (`data/options_portfolio.csv`)

| Column | Description | Example |
|--------|-------------|---------|
| Date | Trade date | 2025-01-10 |
| Cross | Currency pair | EURUSD |
| Expiry | Option expiry date | 2025-04-14 |
| Long/Short | Position direction | Long |
| Call/Put | Option type | Call |
| Strike | Strike price | 1.0500 |
| Notional | Position size | 5000000 |
| Price | Trade premium | 0.0125 |

### Forwards Blotter (`data/forwards_blotter.csv`)

| Column | Description | Example |
|--------|-------------|---------|
| Date | Trade date | 2025-01-10 |
| Cross | Currency pair | EURUSD |
| Buy/Sell | Direction | Buy |
| Notional | Position size | 1000000 |
| Value Date | Settlement date | 2025-04-14 |
| Rate | Trade rate (optional) | 1.0450 |

## Usage

### Main Window
- View aggregated positions by cross/expiry
- Monitor P&L (options, forwards, total)
- Track delta exposure across the book
- Click "Refresh Data" to update market data
- Click "Add Forward" to enter new hedge trades

### Position Detail
- Double-click any row to open detailed view
- See individual option legs with Greeks
- View forward trades for the same expiry
- Add hedging forwards directly from detail view

### Adding Forwards
Example: Buy 1,000,000 EURUSD value date 14/04/2026 @ 1.1701
1. Click "Add Forward"
2. Select currency pair
3. Choose Buy or Sell
4. Enter notional amount
5. Select value date
6. Optionally enter trade rate
7. Click "Add Forward"

## Architecture

```
fx-monitor/
├── main.py                    # Application entry point
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── data/                      # CSV data files
│   ├── options_portfolio.csv
│   └── forwards_blotter.csv
└── src/
    ├── core/                  # Business logic
    │   ├── bloomberg_client.py    # Bloomberg API wrapper
    │   ├── volatility_surface.py  # Vol surface interpolation
    │   ├── pricing_engine.py      # QuantLib pricing
    │   ├── portfolio_manager.py   # Options portfolio
    │   └── forward_manager.py     # Forwards blotter
    ├── gui/                   # User interface
    │   ├── styles.py              # Dark theme styling
    │   ├── main_window.py         # Main application window
    │   ├── detail_window.py       # Position detail view
    │   └── add_forward_dialog.py  # Forward entry dialog
    └── utils/                 # Utilities
        ├── config_loader.py       # YAML config loader
        └── date_utils.py          # Date/tenor utilities
```

## Greeks Calculation

### Forward Delta
Black-Scholes forward delta using the formula:
```
d1 = (ln(F/K) + 0.5 * σ² * T) / (σ * √T)
Delta_call = N(d1)
Delta_put = N(d1) - 1
```

### Vega
Price sensitivity to 1% volatility change, converted to USD and EUR based on the currency pair.

### Gamma 1%
Practical gamma measure: change in delta for a 1% parallel shift in spot and forward rates.

## Volatility Interpolation

### Delta Dimension
- Cubic spline interpolation between 10Δ put, 25Δ put, ATM, 25Δ call, 10Δ call
- Risk reversal and butterfly quotes converted to individual call/put vols

### Time Dimension
- Linear interpolation in variance-time space
- Ensures no-arbitrage condition across tenors
- Extrapolation for expiries beyond 1Y

## Troubleshooting

### Bloomberg Connection Failed
- Ensure Bloomberg Terminal is running and logged in
- Check that Desktop API is enabled (DAPI<GO>)
- Verify firewall allows localhost:8194

### QuantLib Import Error
- On Windows, ensure Visual C++ Redistributable is installed
- Try: `pip install QuantLib-Python` as alternative

### Missing Market Data
- Some crosses may not have all vol surface pillars
- Application uses defaults when data is unavailable

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Disclaimer

This software is for educational and informational purposes only. It is not intended as financial advice. Always verify calculations independently before making trading decisions.
