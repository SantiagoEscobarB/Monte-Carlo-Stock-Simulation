import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t as t_dist
import yfinance as yf


ticker = 'AMZN'
start_date = '2021-01-01'

def import_data(ticker, start_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end='2026-06-15')
    return data

data = import_data(ticker, start_date)
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.title(f'{ticker} Close Price Over Time')
plt.grid(True)
plt.show()

def compute_log_returns(data):
    log_ret = np.log(1 + data['Close'].pct_change())
    return log_ret[1:]

log_returns = compute_log_returns(data)
print(log_returns.tail())

t_df, t_loc, t_scale = t_dist.fit(log_returns)
print(f'Fitted t-distribution: df={t_df:.2f}, loc={t_loc:.6f}, scale={t_scale:.6f}')

plt.figure(figsize=(10, 6))
plt.plot(log_returns, label='Log Returns')
plt.xlabel('Date')
plt.ylabel('Log Returns')
plt.legend()
plt.title(f'{ticker} Log Returns Over Time')
plt.grid(True)
plt.show()

def calculate_volatility(log_returns):
    return log_returns.std()

volatility = calculate_volatility(log_returns)

number_of_simulations = 200
number_of_days = 100
last_price = data['Close'].iloc[-1]

def monte_carlo_simulation(number_of_simulations, number_of_days, last_price, t_df, t_loc, t_scale):
    all_simulated_prices = []

    for _ in range(number_of_simulations):
        price_series = [last_price]
        for _ in range(number_of_days):
            shock = t_dist.rvs(df=t_df, loc=t_loc, scale=t_scale)
            price = price_series[-1] * np.exp(shock)
            price_series.append(price)
        all_simulated_prices.append(price_series)

    return pd.DataFrame(all_simulated_prices).T

simulation_df = monte_carlo_simulation(number_of_simulations, number_of_days, last_price, t_df, t_loc, t_scale)

plt.figure(figsize=(10, 6))
plt.plot(simulation_df, alpha=0.5, linewidth=0.8)
plt.axhline(y=last_price, color='red', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Price')
plt.title(f'Monte Carlo Simulation for: {ticker}')
plt.grid(True)
plt.show()

final_prices = simulation_df.iloc[-1]
price_mean = final_prices.mean()
price_5pct = final_prices.quantile(0.05)
price_95pct = final_prices.quantile(0.95)

plt.figure(figsize=(10, 6))
plt.hist(final_prices, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(x=last_price, color='red', linewidth=2, label=f'Current Price: ${last_price:.2f}')
plt.axvline(x=price_mean, color='green', linewidth=2, linestyle='--', label=f'Expected Price: ${price_mean:.2f}')
plt.axvline(x=price_5pct, color='orange', linewidth=2, linestyle='--', label=f'5th Percentile (VaR): ${price_5pct:.2f}')
plt.axvline(x=price_95pct, color='blue', linewidth=2, linestyle='--', label=f'95th Percentile: ${price_95pct:.2f}')
plt.xlabel(f'Price at Day {number_of_days}')
plt.ylabel('Frequency')
plt.title(f'Distribution of Simulated Prices at Day {number_of_days} for: {ticker}')
plt.legend()
plt.grid(True)
plt.show()


