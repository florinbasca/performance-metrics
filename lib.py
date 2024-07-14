from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import random
import os

import yfinance as yf
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import exchange_calendars as xcals


def getTickerPrice(symbol: str, date: datetime) -> float:
    # This function returns the price of the security 'ticker' at the given 'date'
    # For the purpose of this exercise, assume it returns a random number
    return random.uniform(1, 100)

def get_benchmark_returns(start: date, end: date, benchmark='SPY') -> pd.DataFrame:
  """ Use yahoo finance to get benchmark returns """
  try:
    df = yf.download(benchmark, start=start, end=end+timedelta(days=1), progress=False)
  except e:
    print(e)
    return None
  df['return'] = df['Adj Close'].pct_change()
  return df['return']

def get_trading_days(start: date=None, end: date=None, exchange='XNYS') -> list:
  """
  Returns a list of trading days between start and end.
  Args:
      start (date, optional): Defaults to None.
      end (date, optional): Defaults to None.
      exchange (str, optional): Defaults to NYSE days.
  Returns:
      list: List of trading days
  """
  
  cal = xcals.get_calendar(exchange)
  schedule = cal.schedule
  days = pd.to_datetime(schedule.index).date
  if start is not None:
    days = [day for day in days if day >= start]
  if end is not None:
    days = [day for day in days if day <= end]
  return sorted(days)

def _validate_trades_df(df):
  trades_schema = DataFrameSchema({
    'date':   Column(pa.Date),
    'symbol': Column(pa.String),
    'side':   Column(pa.String, checks=Check.isin(['buy', 'sell'])),
    'size':   Column(pa.Float, nullable=True),  # Nullable to allow missing values that default to 1.0
    'price':  Column(pa.Float, nullable=False),
  })

  validated_df = trades_schema.validate(df)
  if 'size' in validated_df.columns:
    validated_df['size'] = validated_df['size'].fillna(1.0)
  return validated_df

def _per_symbol_calculations(df, symbol, trading_days):
  """ Get pnl and notional per day for each symbol """

  # Aggregate and separate buys and sells
  df = df.groupby(['date', 'side']).agg({
    'size': 'sum',
    'price': lambda x: np.average(x, weights=df.loc[x.index, 'size'])
  }).reset_index()
  df = df.pivot_table(index=['date'], columns='side', values=['size', 'price'], fill_value=0)
  df = df.reindex(trading_days)
  df = df.sort_index()

  df.columns = pd.MultiIndex.from_tuples([(col[0], col[1]) if col[1] in ['buy', 'sell'] else (col[0], col[0]) for col in df.columns])
  for col in [('price', 'buy'), ('price', 'sell'), ('size', 'buy'), ('size', 'sell')]:
    if col not in df.columns:
      df[col] = 0

  # Define eod_price using getTickerPrice function
  df['eod_price'] = df.index.to_frame().apply(lambda row: getTickerPrice(symbol, row['date']), axis=1)

  df[('cumulative', 'buy')] = df[('size', 'buy')].fillna(0).cumsum()
  df[('cumulative', 'sell')] = df[('size', 'sell')].fillna(0).cumsum()
  df['position'] = df[('cumulative', 'buy')] - df[('cumulative', 'sell')]
  df['long'] = df['position'].apply(lambda x: max(x, 0)).abs()
  df['short'] = df['position'].apply(lambda x: min(x, 0)).abs()
  df = df.drop(columns=['cumulative'], axis=1)

  df[('closed', 'cover')] = np.minimum(df[('size', 'buy')], df['short'].shift()).fillna(0)
  df[('closed', 'sell')]  = np.minimum(df[('size', 'sell')], df['long'].shift()).fillna(0)

  # get base price for longs
  df[('base_price', 'long')] = np.nan
  # if long and no new long trades
  condition1 = df['long'].shift().ge(df['long'])
  condition2 = df['long'].shift().eq(0) & df['short'].eq(0)
  condition = df['long'].gt(0) & (condition1 | condition2)
  df.loc[condition, ('base_price', 'long')] = df[('price', 'buy')]
  df[('base_price', 'long')] = df[('base_price', 'long')].ffill()

  # if long and new long trades
  condition1 = df['long'].shift() < df['long']
  condition2 = df['long'].shift() != 0
  condition = df['long'].gt(0) & (condition1 & condition2)
  if condition.any():  # reset base price if new condition
    idx = df.index.get_loc(df[condition].index[0])
    df.iloc[idx:, df.columns.get_loc(('base_price', 'long'))] = np.nan
  df.loc[condition, ('base_price', 'long')] = \
    ((df[('price', 'buy')] * df[('size', 'buy')] + df[('base_price', 'long')].shift()*df['long'].shift()) / df['long'])
  df[('base_price', 'long')] = df[('base_price', 'long')].ffill()
  df.loc[df['long'] <= 0, ('base_price', 'long')] = np.nan

  # get base price for shorts
  df[('base_price', 'short')] = np.nan
  # if short and no new short trades
  condition1 = df['short'].shift().ge(df['short'])
  condition2 = df['short'].shift().eq(0) & df['long'].eq(0)
  condition = df['short'].gt(0) & (condition1 | condition2)
  df.loc[condition, ('base_price', 'short')] = df[('price', 'sell')]
  df[('base_price', 'short')] = df[('base_price', 'short')].ffill()

  # if short and new short trades
  condition1 = df['short'].shift() < df['short']
  condition2 = df['short'].shift() != 0
  condition = df['short'].gt(0) & (condition1 & condition2)
  if condition.any():  # reset base price if new condition
    idx = df.index.get_loc(df[condition].index[0])
    df.iloc[idx:, df.columns.get_loc(('base_price', 'short'))] = np.nan
  df.loc[condition, ('base_price', 'short')] = \
    ((df[('price', 'sell')] * df[('size', 'sell')] + df[('base_price', 'short')].shift()*df['short'].shift()) / df['short'])
  df[('base_price', 'short')] = df[('base_price', 'short')].ffill()
  df.loc[df['short'] <= 0, ('base_price', 'short')] = np.nan

  # pnl
  df['pnl_realized'] = \
     df[('closed', 'sell')]  * (df[('price', 'sell')] - df[('base_price', 'long')].shift().fillna(0)) + \
    -df[('closed', 'cover')] * (df[('price', 'buy')]  - df[('base_price', 'short')].shift().fillna(0))
  df['pnl_unrealized'] = \
     df['long']  * (df['eod_price'] - df[('base_price',  'long')].fillna(0)) + \
    -df['short'] * (df['eod_price'] - df[('base_price', 'short')].fillna(0))
  df['pnl'] = df['pnl_realized'].fillna(0).cumsum() + df['pnl_unrealized'].fillna(0)

  # notional
  df[('notional')] = df['position'] * df['eod_price']

  df = df[['notional', 'pnl']]
  df.columns = df.columns.droplevel(level=1)
  return df

def performance_metrics(trades: pd.DataFrame, starting_equity: float=None, rfr: float=0.05) -> dict:
  """ 
  Calculate performance metrics for a given dataframe.
  Print the results to the console.
  Assumptions:
    - risk-free rate is 5% annualized
    - when starting equity is missing, starting equity is equal to max gross notional during the trading period
    - benchmark is SPY
    - there are no fees

  Args:
    df (pandas.DataFrame):    A dataframe of trades
      date (datetime64[ns]):  The date and time of the trade
      symbol (string):        The ticker symbol of the traded security
      side (string):          Either 'buy' or 'sell'
      size (float, optional): The number of shares traded (default to 1 if not provided)
      price (float):          The price at which the trade was executed
  Returns:
    metrics (dict): dictionary of metrics
  """  

  # validate input
  trades['date'] = pd.to_datetime(trades['date']).dt.date
  trades['size'] = trades['size'].fillna(1).astype(float)
  trades['price'] = trades['price'].astype(float)
  trades = _validate_trades_df(trades)

  # default metrics
  metrics = {
    'calendar_days': pd.NA,
    'total_return': pd.NA,
    'annualized_return': pd.NA,
    'max_drawdown_pct': pd.NA,
    'sharpe_ratio': pd.NA,
    'sortino_ratio': pd.NA,
    'max_leverage': pd.NA,
    'monthly_volatility': pd.NA,
    'win_rate_days': pd.NA,
    'beta': pd.NA,
  }

  if trades.empty:
    return metrics
  
  # get NYSE trading days
  trading_days = get_trading_days(start=trades['date'].min(), end=trades['date'].max())
  trading_days.append(min(trading_days) - timedelta(days=1))  # need one empty date before any trading happens
  trading_days = pd.Index(sorted(trading_days), name='date')

  # get notional and pnl by symbol and date, this is the core of the metric calculations
  with ThreadPoolExecutor(max_workers=max(1, os.cpu_count()-2)) as executor:
    dfs = [executor.submit(_per_symbol_calculations, df, symbol, trading_days).result()
      for symbol, df in trades.groupby('symbol')]
  df = pd.concat(dfs)
  summary_df = df.groupby(level='date').agg({
    'notional': lambda x: x.abs().sum(),  # gross notional
    'pnl': 'sum'
  }).rename(columns={'notional': 'gross_notional', 'pnl': 'cummulative_pnl'})
  summary_df['net_notional'] = df.groupby(level='date')['notional'].sum()
  df = summary_df

  # calculate returns, deal with a potential lack of starting_equity, make sure leverage is at most 1x, imperfect, but better than nothing
  if starting_equity is None:
    starting_equity = df['gross_notional'].max()
  df['equity'] = starting_equity + df['cummulative_pnl']
  df['return'] = df['equity'].pct_change().fillna(0)

  # total return
  total_return = df['equity'].iloc[-1] / df['equity'].iloc[0] - 1
  
  # annualized return
  calendar_trading_days = (trading_days[-1] - trading_days[1]).days +1
  annualized_return = (1 + total_return) ** (365 / calendar_trading_days) - 1

  # max drawdown
  df['drawdown'] = (df['equity'] - df['equity'].cummax()) / df['equity'].cummax()
  max_drawdown_pct = df['drawdown'].min()
  
  # annualized sharpe ratio
  sharpe_ratio = (annualized_return-rfr) / (df['return'].std() * np.sqrt(252))

  # annualized sortino ratio
  sortino_ratio = (annualized_return-rfr) / (df['return'][df['return'] < 0].std() * np.sqrt(252))
  
  # max leverage
  max_leverage = (df['gross_notional'] / df['equity']).max()

  # monthly volatility
  monthly_volatility = df['return'].std() * np.sqrt(252) / np.sqrt(12)

  # win_rate_days
  win_rate_days = len(df[df['return'] > 0]) / (len(df)-1)

  # beta
  benchmark = get_benchmark_returns(trading_days[0], trading_days[-1])
  if benchmark is not None:
    df['benchmark'] = benchmark
    df['benchmark'] = df['benchmark'].fillna(0)
    covariance = np.cov(df['return'].iloc[1:], df['benchmark'].iloc[1:])[0, 1]  # ignore data in the first row as returns are missing
    beta = covariance / np.var(df['benchmark'])
  else:
    beta = pd.NA

  metrics['calendar_days']      = calendar_trading_days
  metrics['total_return']       = total_return
  metrics['annualized_return']  = annualized_return
  metrics['max_drawdown_pct']   = max_drawdown_pct
  metrics['sharpe_ratio']       = sharpe_ratio
  metrics['sortino_ratio']      = sortino_ratio
  metrics['max_leverage']       = max_leverage
  metrics['monthly_volatility'] = monthly_volatility
  metrics['win_rate_days']      = win_rate_days
  metrics['beta']               = beta
  return metrics
