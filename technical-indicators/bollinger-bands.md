# Bollinger Bands

Bollinger Bands are a technical analysis tool developed by John Bollinger in the 1980s. They consist of a middle band (a simple moving average) with an upper and lower band placed at a specific number of standard deviations away from the middle band, creating a dynamic price envelope that expands and contracts based on market volatility.

## Historical Background

### Development and Creator

Bollinger Bands were developed by John Bollinger, a renowned technical analyst and market technician, in the early 1980s. Prior to Bollinger's innovation, traders had been using fixed percentage bands around moving averages, but Bollinger recognized that market volatility is dynamic rather than static.

Bollinger formally introduced his bands in the 1980s through his market newsletter and later detailed the concept in his 2001 book "Bollinger on Bollinger Bands." He received the Market Technicians Association's Annual Award in 2005 for his significant contribution to technical analysis.

The indicator has since become one of the most widely recognized and used technical analysis tools, available on virtually all trading platforms and charting software.

## Mathematical Foundation

### Formula and Calculation

Bollinger Bands consist of three components:

1. **Middle Band (MB)**: A simple moving average (SMA) of the price, typically using a 20-period calculation
2. **Upper Band (UB)**: The middle band plus a specified number of standard deviations (typically 2)
3. **Lower Band (LB)**: The middle band minus the same number of standard deviations

The mathematical formulas are:

```
Middle Band = SMA(n)
Upper Band = SMA(n) + (k × σ[n])
Lower Band = SMA(n) - (k × σ[n])
```

Where:
- SMA(n) is the simple moving average over n periods
- k is the multiplier for the standard deviation (typically 2)
- σ[n] is the standard deviation of price over the same n periods

### Step-by-Step Calculation Example

Let's calculate the 20-day Bollinger Bands with 2 standard deviations for a hypothetical stock:

1. Calculate the 20-day simple moving average (SMA) of closing prices
   - Assume the 20-day SMA = $100

2. Calculate the standard deviation of closing prices over the same 20-day period
   - Assume the 20-day standard deviation = $5

3. Calculate the upper and lower bands
   - Upper Band = $100 + (2 × $5) = $110
   - Lower Band = $100 - (2 × $5) = $90

As new price data becomes available, recalculate these values to create the dynamic bands.

### Parameters and Customization

While the standard parameters are 20 periods and 2 standard deviations, they can be adjusted for different market conditions and timeframes:

- **Period Length (n)**: Typically 20, but can range from 10 (more responsive) to 50 (smoother)
- **Standard Deviation Multiplier (k)**: Typically 2, but can be adjusted:
  - Higher values (2.5-3) create wider bands with fewer touches/breaks
  - Lower values (1.5-1.8) create tighter bands with more frequent touches/breaks
- **Moving Average Type**: While typically an SMA, some traders use an exponential moving average (EMA) for the middle band to be more responsive to recent price changes

## Interpretation and Analysis

### Basic Interpretation

Bollinger Bands provide several types of signals and insights:

#### Volatility Measurement
- **Band Width**: The distance between the upper and lower bands indicates market volatility
  - Widening bands suggest increasing volatility
  - Narrowing bands suggest decreasing volatility

#### Price Containment
- Approximately 90% of price action occurs within the bands when using 2 standard deviations
- Prices touching or exceeding the bands are statistically significant events

#### Mean Reversion Tendency
- Prices tend to return to the middle band (the moving average) over time
- Touches of the outer bands often suggest potential price reversals toward the middle

### Advanced Analysis Techniques

#### Bollinger Band Squeeze
A "squeeze" occurs when volatility falls to a low level, as indicated by the bands narrowing:
- Often precedes significant price movements
- Direction of the subsequent move is not predicted by the squeeze itself
- The longer and tighter the squeeze, the more significant the potential breakout

#### Bollinger Band Breakouts
- **Genuine Breakouts**: Strong price movements beyond the bands that continue in the direction of the breakout
- **False Breakouts**: Brief excursions beyond the bands that quickly reverse

#### Double Bottoms and Tops
- **Double Bottom (W-Bottom)**: First bottom touches or exceeds the lower band, second bottom is higher than the first and stays within the bands, suggesting stronger support
- **Double Top (M-Top)**: First top touches or exceeds the upper band, second top is lower than the first and stays within the bands, suggesting weaker momentum

#### Bollinger Band "Walking"
- When prices consistently touch or move along one of the bands while trending:
  - "Walking the upper band": Strong uptrend
  - "Walking the lower band": Strong downtrend

#### Percent Bandwidth (%B)
A normalized indicator showing where price is relative to the bands:
```
%B = (Price - Lower Band) / (Upper Band - Lower Band)
```
- %B above 1.0: Price above upper band
- %B at 1.0: Price at upper band
- %B at 0.5: Price at middle band
- %B at 0.0: Price at lower band
- %B below 0.0: Price below lower band

### The Bollinger Band Rules

John Bollinger formulated several rules for using his bands effectively:

1. **Sharp price movements tend to occur after the bands tighten**
2. **When prices move outside the bands, continuation in the current direction is implied**
3. **Bottoms and tops made outside the bands followed by bottoms and tops inside the bands call for reversals in the trend**
4. **A move that originates at one band tends to go to the opposite band**
5. **Band width provides a relative definition of high and low volatility**

## Applications in Trading

### Trading Strategies

#### Bollinger Band Bounce Strategy
A mean reversion approach:
1. Wait for price to touch or slightly penetrate the outer bands
2. Look for confirming signals (candlestick patterns, momentum indicators)
3. Enter a position anticipating a move back toward the middle band
4. Set stop-loss beyond the recent extreme
5. Target the middle band or opposite band for profit-taking

#### Bollinger Band Breakout Strategy
A momentum approach:
1. Wait for the Bollinger Band squeeze (narrowing of bands)
2. Enter when price decisively breaks through either band after the squeeze
3. Use volume confirmation for higher probability trades
4. Set stop-loss at the middle band or recent swing point
5. Target 1-2 times the recent Bollinger Band width for profit objectives

#### Bollinger Band Trend Following Strategy
Using bands to identify and follow trends:
1. Identify trend direction using the middle band slope and price position
2. In uptrends, buy pullbacks to the middle band
3. In downtrends, sell rallies to the middle band
4. Exit when price crosses the opposite band or when the trend changes

#### Double Bottom/Double Top Strategy
Trading reversal patterns:
1. Identify W-Bottom or M-Top formations with proper Bollinger Band relationship
2. Wait for the "trigger" (middle peak of W or valley of M to be broken)
3. Enter in the direction of the breakout
4. Set stop-loss at the recent extreme
5. Target at least the height of the formation

### Combining Bollinger Bands with Other Indicators

Bollinger Bands work well when combined with other technical tools:

- **Bollinger Bands with RSI**:
  - Use RSI to confirm overbought/oversold conditions at band touches
  - Look for RSI divergence when price reaches the bands
  - Higher probability signals when both tools align

- **Bollinger Bands with MACD**:
  - MACD crossovers that coincide with band touches or breakouts
  - MACD histogram direction change confirming potential reversals from the bands
  - MACD divergence supporting Bollinger Band pattern signals

- **Bollinger Bands with Volume Indicators**:
  - Volume expansion confirming Bollinger Band breakouts
  - Volume contraction during squeezes reinforcing potential energy building
  - Volume profile supporting key levels near the bands

- **Bollinger Bands with Stochastic**:
  - Stochastic overbought/oversold aligning with band touches
  - Stochastic crossovers confirming potential reversals from the bands
  - Especially effective in ranging markets

## Bollinger Bands Across Different Markets and Timeframes

### Market-Specific Considerations

- **Stocks**: Standard 20,2 settings work well for most stocks; higher volatility stocks may benefit from wider bands (2.5-3 standard deviations)
- **Forex**: Often effective with slightly narrower bands (1.8-2 standard deviations) due to lower overall volatility in major pairs
- **Cryptocurrencies**: May require wider bands (2.5-3 standard deviations) or longer periods due to higher volatility
- **Commodities**: Works well with standard settings but benefits from market-specific adjustments based on historical volatility

### Timeframe Considerations

- **Intraday (Minutes, Hours)**: Shorter periods (10-15) for more signals; useful for scalping and day trading
- **Daily**: Standard 20,2 settings work well for swing trading
- **Weekly**: Provides stronger signals for longer-term position trades; fewer but more significant signals
- **Monthly**: Very long-term analysis; signals are rare but highly significant for major trend changes

## Advantages and Limitations

### Advantages

- **Adaptive to Volatility**: Automatically adjusts to changing market conditions
- **Multiple Applications**: Useful for identifying volatility, potential reversals, trends, and breakouts
- **Visual Clarity**: Provides clear visual representation of price extremes
- **Versatility**: Effective across different markets and timeframes
- **Statistical Foundation**: Based on standard deviation, giving statistical significance to signals

### Limitations

- **Lagging Element**: As with all indicators based on moving averages, contains some lag
- **False Signals**: Band touches don't always lead to reversals, and breakouts can fail
- **Normal Distribution Assumption**: Standard deviation assumes normal distribution, which isn't always true for financial markets
- **Parameter Sensitivity**: Results can vary based on the chosen period and standard deviation settings
- **Requires Context**: Most effective when used with other indicators and within a comprehensive trading plan

## Best Practices

- Confirm Bollinger Band signals with other indicators or price action
- Adjust parameters based on the specific security's volatility and your trading timeframe
- Remember that bands define relative high and low prices, not absolute buy and sell signals
- Pay attention to band width for volatility insights, not just price position relative to the bands
- Use the middle band to help identify the overall trend direction
- Be cautious of false breakouts, especially without volume confirmation
- Consider multiple timeframe analysis for more robust signals
- Backtest your specific Bollinger Band strategy before live trading

## Programming Implementation

Python implementation of Bollinger Bands calculation:

```python
import pandas as pd
import numpy as np

def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands
    
    Parameters:
    data (DataFrame or Series): Price data, typically close prices
    window (int): Look-back period for the moving average
    num_std (float): Number of standard deviations for the bands
    
    Returns:
    DataFrame with middle band, upper band, and lower band
    """
    # Calculate middle band (simple moving average)
    middle_band = data.rolling(window=window).mean()
    
    # Calculate the standard deviation
    std_dev = data.rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    # Calculate %B
    percent_b = (data - lower_band) / (upper_band - lower_band)
    
    # Calculate bandwidth
    bandwidth = (upper_band - lower_band) / middle_band
    
    return pd.DataFrame({
        'Middle Band': middle_band,
        'Upper Band': upper_band,
        'Lower Band': lower_band,
        '%B': percent_b,
        'Bandwidth': bandwidth
    })
```

## Notable Research and Literature

- John Bollinger's book, "Bollinger on Bollinger Bands" (2001)
- John Bollinger's articles in "Technical Analysis of Stocks & Commodities" magazine
- Alexander Elder's analysis in "Trading for a Living" (1993)
- Perry Kaufman's research in "Trading Systems and Methods" (various editions)
- Academic studies testing Bollinger Bands effectiveness across various markets
- Lorna Reichert's work on Bollinger Band pattern recognition
- David Stendahl's research on optimizing Bollinger Band parameters

## Related Indicators

- [Moving Averages](./moving-averages.md)
- [Keltner Channels](./keltner-channels.md)
- [Donchian Channels](./donchian-channels.md)
- [Average True Range (ATR)](./atr.md)
- [Relative Strength Index (RSI)](./rsi.md)
- [Moving Average Convergence Divergence (MACD)](./macd.md)
- [Stochastic Oscillator](./stochastic-oscillator.md) 