from datetime import date
from lib import performance_metrics, getTickerPrice
import pandas as pd
import numpy as np


def sample():
  df = pd.DataFrame({
    'date': ['2024-01-04', '2024-01-08', '2024-01-08', '2024-01-08', '2024-01-11', '2024-01-12'],
    'symbol': ['AAPL', 'AAPL', 'AAPL', 'NVDA', 'NVDA', 'AAPL'],
    'side':    ['buy', 'sell',  'sell', 'buy',  'sell', 'buy'],
    'size':    [   2,       2,      1,      1,     1,     1],
    'price':   [  10,    10.1,     11,     10,    10,    11]
  })
  metrics = performance_metrics(df)
  print(metrics)


def pelosi_trades():
  raw = pd.read_csv('testData.csv')

  # preprocess
  raw = raw.rename(columns={
    'transactionDate': 'date',
    'ticker': 'symbol',
    })
  raw = raw.sort_values('date')
  raw['price'] = raw.apply(lambda row: getTickerPrice(row['symbol'], row['date']), axis=1)

  all_df = []
  # filter for rows that don't make sense
  # can not have a sale after a "Sale (full)"
  # can not start with a salse
  for symbol, df in raw.groupby('symbol'):

    df['include'] = True
    purchase_seen = False
    full_sale = False
    for idx, row in df.iterrows():
      if full_sale or not purchase_seen:
        df.loc[idx, 'include'] = False

      if row['type'] == 'Sale (Full)':
        full_sale = True
      if row['type'] == 'Purchase':
        purchase_seen = True
        full_sale = False
        df.loc[idx, 'include'] = True
    df = df[df['include']].copy()
    df['amount'] = df['amount'].str.split('$').str[-1].str.replace(',','').astype(float)
    df['size'] = df['amount'] / df['price']
    df['side'] = df['type'].apply(lambda x: 'buy' if x == 'Purchase' else 'sell')

    # adjust sales for partials - assume only 1/2 of position is sold during a partial, all of position is sold when full
    position = 0
    rows = []
    for idx, row in df.iterrows():
      size = row['size']
      side = row['side']
      if side == 'sell':
        if row['type'] == 'Sale (Partial)':
          if size > position:
            size = position / 2
        elif row['type'] == 'Sale (Full)':
          size = position
        position -= size
      else:
        position += size
      row['size'] = size
      rows.append(row)
    df = pd.DataFrame(rows)
    all_df.append(df[['date', 'symbol', 'side', 'size', 'price']])
  df = pd.concat(all_df, ignore_index=True)      

  metrics = performance_metrics(df)
  print(metrics)


if __name__ == '__main__':
  pelosi_trades()