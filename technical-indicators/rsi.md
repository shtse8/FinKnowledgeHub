# Relative Strength Index (RSI)

The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. It oscillates between 0 and 100 and is typically used to identify overbought or oversold conditions in a traded security.

## Historical Background

### Development and Creator

The Relative Strength Index was developed by J. Welles Wilder Jr. and introduced in his 1978 book, "New Concepts in Technical Trading Systems." Wilder was a mechanical engineer who later became a real estate developer before turning to technical analysis. He is also known for developing several other technical indicators including the Average Directional Index (ADX), the Parabolic SAR, and the Average True Range (ATR).

Wilder originally designed the RSI for analyzing commodities, but it has since become one of the most popular indicators for all financial markets, including stocks, forex, and cryptocurrencies.

## Mathematical Foundation

### Formula and Calculation

The RSI is calculated using the following formula:

```
RSI = 100 - (100 / (1 + RS))
```

Where:
- RS (Relative Strength) = Average Gain / Average Loss
- Average Gain = Sum of gains over specified period / period length
- Average Loss = Sum of losses over specified period / period length (expressed as a positive value)

The standard period used in RSI calculations is 14 periods, though traders may use shorter periods for more sensitivity or longer periods for less sensitivity.

### Step-by-Step Calculation Example

Let's calculate a 14-period RSI for a hypothetical stock:

1. Calculate the price change for each period (typically each day)
2. Separate the positive changes (gains) and negative changes (losses)
3. Calculate the average gain and average loss over 14 periods
4. Calculate the Relative Strength (RS) as average gain divided by average loss
5. Apply the RSI formula: RSI = 100 - (100 / (1 + RS))

For example, if:
- Average 14-day gain = $0.80
- Average 14-day loss = $0.50
- RS = 0.80/0.50 = 1.6
- RSI = 100 - (100 / (1 + 1.6)) = 100 - 38.46 = 61.54

### Smoothing Modification

While the initial calculation involves simple averages, subsequent calculations use a smoothing method:

- First 14 periods: Simple averages of gains and losses
- Subsequent periods:
  * Average Gain = ((Previous Average Gain × 13) + Current Gain) / 14
  * Average Loss = ((Previous Average Loss × 13) + Current Loss) / 14

This smoothing approach prevents significant price changes from dropping out of the calculation all at once.

## Interpretation and Analysis

### Basic Interpretation

The RSI provides several types of trading signals:

#### Overbought and Oversold Levels
- **Overbought Condition**: RSI reading above 70 (originally 80 in Wilder's work)
- **Oversold Condition**: RSI reading below 30 (originally 20 in Wilder's work)

These thresholds suggest potential price reversals, though they are not absolute buy or sell signals. Markets can remain overbought or oversold for extended periods during strong trends.

#### Centerline Crossovers
- **Bullish Signal**: RSI crosses above the 50 level, suggesting increasing bullish momentum
- **Bearish Signal**: RSI crosses below the 50 level, suggesting increasing bearish momentum

#### Divergences
- **Bullish Divergence**: Price makes a lower low, but RSI makes a higher low, suggesting diminishing downward momentum
- **Bearish Divergence**: Price makes a higher high, but RSI makes a lower high, suggesting diminishing upward momentum

Divergences are often considered stronger signals than overbought/oversold conditions alone.

### Advanced Analysis Techniques

#### Failure Swings
A "failure swing" occurs without the RSI crossing the centerline:
- **Bullish Failure Swing**: RSI falls below 30, bounces, pulls back but stays above 30, then breaks above the bounce high
- **Bearish Failure Swing**: RSI rises above 70, drops, rebounds but stays below 70, then breaks below the drop low

#### RSI Trendline Analysis
- Drawing trendlines on the RSI itself can provide signals, often before they appear in price
- Breaking a downtrend line in the RSI while in oversold territory is a strong bullish signal
- Breaking an uptrend line in the RSI while in overbought territory is a strong bearish signal

#### RSI Range Shifts
- During strong uptrends, RSI often oscillates between 40-80 rather than 30-70
- During strong downtrends, RSI often oscillates between 20-60 rather than 30-70
- Recognizing these range shifts helps avoid premature signals

### Customized RSI Variations

The standard 14-period RSI can be modified for different market conditions or trading preferences:

- **Short-Term RSI**: 9-period or 7-period for day trading or increased sensitivity
- **Long-Term RSI**: 21-period or 25-period for reduced noise and longer-term signals
- **Modified Threshold Levels**: 80/20 for stronger signals or 60/40 for earlier signals
- **Cutler's RSI**: A variation that uses simple moving averages rather than Wilder's smoothing method
- **Connors RSI**: A composite indicator that incorporates traditional RSI with rate-of-change and ranking components

## Applications in Trading

### Trading Strategies

#### RSI Reversal Strategy
A basic strategy using overbought/oversold levels for mean reversion:
1. Enter long when RSI drops below 30 and then rises back above it
2. Enter short when RSI rises above 70 and then drops back below it
3. Use stop-loss orders based on recent price action or volatility
4. Target the median range or opposing threshold for profit-taking

#### RSI Divergence Strategy
Trading based on divergence between price and RSI:
1. Identify bullish or bearish divergence
2. Confirm with RSI crossing back through 30 or 70 level
3. Enter position in the direction suggested by the divergence
4. Set stop-loss beyond recent swing high/low
5. Target next significant support/resistance level

#### RSI Trend Following Strategy
Using centerline crossovers for trend following:
1. Enter long when RSI crosses above 50 in an uptrend
2. Enter short when RSI crosses below 50 in a downtrend
3. Combine with longer-term trend indicators for confirmation
4. Exit when RSI indicates overbought/oversold conditions

#### RSI Swing Rejection Strategy
Based on failure swings and range shifts:
1. In an uptrend, buy when RSI tests but doesn't break below 40, then turns up
2. In a downtrend, sell when RSI tests but doesn't break above 60, then turns down
3. Confirm with candlestick patterns or price action
4. Set tight stops below/above the reaction low/high

### Combining RSI with Other Indicators

RSI is most effective when combined with other indicators:

- **RSI with Moving Averages**:
  - Use long-term moving averages to establish trend direction
  - Take RSI signals only in the direction of the larger trend
  - Moving average crossovers can confirm RSI signals

- **RSI with MACD**:
  - Combine RSI overbought/oversold signals with MACD crossovers
  - Look for convergence between both indicators for stronger signals
  - MACD can confirm RSI divergences

- **RSI with Support/Resistance Levels**:
  - RSI signals have higher probability when they occur at key price levels
  - Use price levels to determine stop-loss and profit targets
  - Watch for price breakouts confirmed by RSI momentum

- **RSI with Volume Indicators**:
  - Volume can confirm or refute RSI signals
  - Strong volume on RSI reversals increases signal reliability
  - Volume divergence can complement RSI divergence

## RSI Across Different Markets and Timeframes

### Market-Specific Considerations

- **Stocks**: Standard RSI parameters work well for most stocks
- **Forex**: May require adjustments for 24-hour trading and different volatility patterns
- **Cryptocurrencies**: Often effective but may need higher thresholds (75/25) due to higher volatility
- **Commodities**: Works well with standard parameters but may benefit from market-specific adjustments

### Timeframe Considerations

- **Intraday (Minutes, Hours)**: Shorter RSI periods (7-9) often used; more signals but higher noise
- **Daily**: Most common timeframe, standard 14-period RSI works well
- **Weekly**: Provides stronger signals for longer-term trends; fewer but more significant trades
- **Monthly**: Very long-term trend identification; signals are rare but highly significant

## Advantages and Limitations

### Advantages

- **Simplicity**: Easy to calculate and interpret
- **Visual Clarity**: Bounded oscillator making overbought/oversold conditions clear
- **Versatility**: Effective across different markets and timeframes
- **Multiple Signal Types**: Offers various ways to analyze market conditions
- **Leading Indicator**: Can provide early warning of potential reversals

### Limitations

- **False Signals**: Can generate false signals in strongly trending markets
- **Subjectivity in Divergence**: Identifying valid divergences requires experience
- **Lagging Element**: Despite being a leading indicator, still incorporates past price data
- **Whipsaws**: Can produce frequent signals in choppy markets
- **No Context**: Doesn't account for fundamental factors or broader market conditions

## Best Practices

- Use RSI in conjunction with other indicators and price action
- Adjust overbought/oversold thresholds based on market conditions
- Be cautious of overbought/oversold signals in strong trends
- Pay special attention to divergences, especially when they occur at significant price levels
- Consider different timeframes for confirmation (higher timeframe RSI supporting lower timeframe signals)
- Combine RSI analysis with support/resistance levels and chart patterns
- Remember that no single indicator, including RSI, is right 100% of the time

## Programming Implementation

Python implementation of RSI calculation:

```python
import pandas as pd
import numpy as np

def calculate_rsi(data, period=14):
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi
```

## Notable Research and Literature

- J. Welles Wilder's original work in "New Concepts in Technical Trading Systems" (1978)
- Constance Brown's analysis of RSI in "Technical Analysis for the Trading Professional" (1999)
- Andrew Cardwell's research on RSI positive and negative reversals
- Thomas Bulkowski's statistical studies on RSI effectiveness in "Encyclopedia of Chart Patterns"
- Academic studies testing RSI performance across various markets and time periods

## Related Indicators

- [Moving Averages](./moving-averages.md)
- [Moving Average Convergence Divergence (MACD)](./macd.md)
- [Stochastic Oscillator](./stochastic-oscillator.md)
- [Commodity Channel Index (CCI)](./cci.md)
- [Average Directional Index (ADX)](./adx.md)
- [Williams %R](./williams-percent-r.md) 