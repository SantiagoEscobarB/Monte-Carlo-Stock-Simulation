# Monte Carlo Stock Price Simulation

Simulates future stock price paths using Monte Carlo methods with a Student's t-distribution to account for the fat tails observed in real market returns.

## What it does

1. **Downloads historical price data** via `yfinance` for a given ticker and start date
2. **Computes log returns** and fits a Student's t-distribution to them using MLE — capturing the actual tail behavior of the stock rather than assuming normality
3. **Runs Monte Carlo simulations** generating N price paths over D trading days, where each daily shock is drawn from the fitted t-distribution
4. **Produces three charts:**
   - Historical close price over time
   - Historical log returns over time
   - Simulated price paths (one line per simulation)
   - Distribution of simulated prices at the final day, with markers for the current price, expected price, and 5th/95th percentiles

## Why Student's t instead of Normal

The standard Black-Scholes model assumes log returns are normally distributed, which underestimates the probability of extreme moves (fat tails / leptokurtosis). Real stock returns crash and spike more often than a normal distribution predicts.

The Student's t-distribution adds a degrees-of-freedom parameter (ν) that controls tail heaviness. A lower ν means fatter tails. The script fits ν directly from historical data, so the tail behavior is calibrated to the actual stock being simulated.

## Setup

```bash
# Create and activate the virtual environment
python -m venv monte-carlo
.\monte-carlo\Scripts\Activate.ps1

# Install dependencies
pip install numpy pandas matplotlib seaborn scipy yfinance setuptools
```

## Usage

Edit the parameters at the top of `monte-carlo.py`:

```python
ticker = 'AMZN'          # Any ticker supported by yfinance
start_date = '2021-01-01'

number_of_simulations = 200   # Number of price paths to simulate
number_of_days = 100          # Forecast horizon in trading days
```

Then run:

```bash
python monte-carlo.py
```

## Output

The console prints the fitted t-distribution parameters:

```
Fitted t-distribution: df=3.21, loc=0.000412, scale=0.009871
```

A low `df` (e.g. 3–6) confirms the stock has significant fat tails. Values above ~30 approximate the normal distribution.

The final histogram shows:

| Line | Meaning |
|---|---|
| Red | Current (last known) price |
| Green | Expected price (mean of all simulations) |
| Orange | 5th percentile — VaR-style downside bound |
| Blue | 95th percentile — upside bound |

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `matplotlib` | Plotting |
| `seaborn` | Plot styling |
| `scipy` | t-distribution fitting (MLE) and sampling |
| `yfinance` | Historical stock data |
| `setuptools` | `distutils` compatibility shim for Python 3.12+ |
