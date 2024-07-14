# performance-metrics
SFIO Assessment

Task 1:
Calculate most important 10 metrics for a strategy based on trades - use function `lib.performance_metrics()`
- calendar_days:      total calendar days from the first trade to the last
- total_return:       total return over all of the trading period
- annualized_return:  total return annualized
- max_drawdown_pct:   percent max drawdown
- sharpe_ratio:       annualized sharpe ratio
- sortino_ratio:      annualized sortino ratio
- max_leverage:       maximum leverage
- monthly_volatility: monthly volatility
- win_rate_days:      percent of winning days vs total trading days
- beta:               beta to benchmark (SPY by default)

The metrics were chosen based on their general use and application to multiple types of strategies. However each strategy has it's own particularity and will require custom metrics. Not included here are slippage and fees, which are extremely important to include if we want an optimal Sharpe at higher leverage.

Returns in both long-only or long/short strategies are determined by comparing profit and loss (PnL) against equity. This task does not specify a required input for initial equity, but the function assumes that starting equity equals the maximum gross exposure during the trading period as a default value in case it is not provided. In practice, there is always an initial equity, and depending on the account configuration, there is also a maximum leverage limit set by the broker.

Assumptions considered:
  - The risk-free rate is set at 5% annualized.
  - If starting equity is not provided, it defaults to the maximum gross notional over the trading period.
  - SPY serves as the benchmark.
  - No transaction fees are applied.

Task 2. Pelosi trades
- Data is incomplete and most likely inaccurate
  - contains only equity trades
  - size information is absent
  - amount represented as a range of dollars
  - trade sequencing errors (eg sales occur after a full sale)
- The function `run.pelosi_trades()` attempts to interpret these trades under the given constraints:
  - Assumes quantities based on the highest amount in the range divided by a daily price (price generated randomly)
  - Treats partial sales as full if they don't result in short positions; otherwise, only half of the position is sold
  - Interprets full sales as liquidation of the entire position
- Results: 
  - Metrics are based on random prices, hence may vary significantly with each execution
  - calendar_days: 719          - about 2 years of trading
  - total_return': 0.065        - quite low considering Pelosi's past performance
  - annualized_return': 0.032   - same but annualized
  - max_drawdown_pct': -0.72    - very high max drawdown this is a 72% loss
  - sharpe_ratio': -0.0047      - very low Sharpe ratio, typical SPY benchmark is around .5
  - sortino_ratio': -0.0090     - very low Sortino ratio, typical SPY benchmark is around .7
  - max_leverage': 2.35         - high, but this was also done with a very loose starting equity assumption
  - monthly_volatility': 1.043  - about 100% monthly volatility, very high
  - win_rate_days': 0.493       - ratio of win days vs all trading days, it's below .5, which is not great
  - beta': -0.086               - surprisingly low correlation to SPY


## Installation
Preinstallation of the following libraries is required:
- pandas
- numpy
- exchange_calendars
- yfinance
- pandera

## Usage
> check `run.py`

## How to run
```
python run.py
```

## How to run tests
Unit tests not yet implemented


## Contact
- Florin Basca florin.basca@yahoo.com
- Github repo: https://github.com/florinbasca/weekdays

