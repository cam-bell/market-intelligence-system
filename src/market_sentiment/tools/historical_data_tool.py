import os
import requests
import datetime as dt
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from typing import Optional

# --- Pydantic Output Model for the Tool ---
class RiskFeatureSet(BaseModel):
    """Derived risk features (the B-Model) calculated from historical OHLCV data."""
    symbol: str = Field(description="The ticker symbol of the asset.")
    realized_vol_21d_ann: float = Field(
        description="Annualized Realized Volatility over the last 21 trading days (252-day convention), expressed as a decimal (e.g., 0.25 for 25%)."
    )
    return_5d: float = Field(description="5-day simple return (e.g., 0.01 for +1%).")
    return_21d: float = Field(description="21-day simple return (e.g., -0.05 for -5%).")
    volume_zscore_20d: float = Field(
        description="The Z-score of today's volume compared to the 20-day average volume. High positive score indicates abnormal volume."
    )
    max_drawdown_21d: Optional[float] = Field(
        description="The largest peak-to-trough decline over the last 21 days, expressed as a negative decimal (e.g., -0.10 for 10% drop)."
    )

# --- The Quant Calculation Functions ---

def calculate_risk_features(df: pd.DataFrame, current_volume: int) -> RiskFeatureSet:
    """
    Calculates the derived risk features from the historical OHLCV DataFrame.
    Assumes the DataFrame has a 'close' column and is sorted by date ascending.
    """
    if df.empty or len(df) < 21:
        # Cannot calculate 21-day metrics without enough data
        return None
    
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 1. Realized Volatility (21-day, Annualized)
    # Volatility is STDEV of log returns. Annualization factor is sqrt(252 trading days / window size)
    DAILY_TRADING_DAYS = 252
    VOL_WINDOW = 21
    
    realized_vol_21d = df['log_return'].iloc[-VOL_WINDOW:].std()
    realized_vol_21d_ann = realized_vol_21d * np.sqrt(DAILY_TRADING_DAYS)

    # 2. Momentum & Returns
    current_close = df['close'].iloc[-1]
    
    return_5d = (current_close / df['close'].iloc[-5] - 1) if len(df) >= 5 else 0.0
    return_21d = (current_close / df['close'].iloc[-21] - 1) if len(df) >= 21 else 0.0
    
    # 3. Volume Z-Score (20-day)
    VOL_Z_WINDOW = 20
    avg_volume = df['volume'].iloc[-VOL_Z_WINDOW:].mean()
    std_volume = df['volume'].iloc[-VOL_Z_WINDOW:].std()
    
    # Use the current snapshot volume for the Z-score calculation
    volume_zscore_20d = (current_volume - avg_volume) / std_volume if std_volume > 0 else 0.0

    # 4. Max Drawdown (21-day)
    window_prices = df['close'].iloc[-VOL_WINDOW:]
    peak = window_prices.cummax()
    drawdown = (window_prices / peak) - 1.0
    max_drawdown_21d = drawdown.min()
    
    return RiskFeatureSet(
        symbol=df['symbol'].iloc[-1],
        realized_vol_21d_ann=realized_vol_21d_ann,
        return_5d=return_5d,
        return_21d=return_21d,
        volume_zscore_20d=volume_zscore_20d,
        max_drawdown_21d=max_drawdown_21d,
    )

# --- The CrewAI Tool Class ---
class HistoricalDataTool(BaseTool):
    name: str = "Historical Data Tool"
    description: str = (
        "Fetches historical OHLCV data (up to 21 days) to compute derived risk "
        "features such as annualized realized volatility, rolling returns, and volume Z-score. "
        "Required for Risk Regime identification."
    )

    def _run(self, ticker: str, current_session_volume: int) -> RiskFeatureSet | str:
        """
        Fetches EOD data and calculates derived features.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL').
            current_session_volume (int): The current day's volume from the MarketSnapshot.

        Returns:
            RiskFeatureSet: The structured derived risk metrics.
        """
        ACCESS_KEY = os.environ.get("MARKETSTACK_API_KEY")
        BASE_URL = "https://api.marketstack.com/v2/eod"
        
        # Request enough data to calculate 21-day metrics, plus a buffer
        date_from = (dt.date.today() - dt.timedelta(days=40)).strftime('%Y-%m-%d')
        
        params = {
            'access_key': ACCESS_KEY,
            'symbols': ticker,
            'date_from': date_from,
            'limit': 40, # Get a max of 40 days to ensure 21 trading days are covered
            'sort': 'ASC' # Important for time-series calculations
        }
        
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data_list = response.json().get('data', [])
            
            if not data_list:
                return f"Error: Not enough historical EOD data returned for {ticker}."

            # Convert to DataFrame for calculation
            df = pd.DataFrame(data_list).sort_values(by='date', ascending=True)
            
            # Ensure required columns are present and typed correctly
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

            # Drop any row with missing close/volume after conversion
            df = df.dropna(subset=['close', 'volume'])

             # Add symbol column if not present
            if 'symbol' not in df.columns:
                df['symbol'] = ticker
            
            # Add the current session's EOD data placeholder for volume z-score calculation
            # We assume the last historical close is the prev_close for today
            df_today = pd.DataFrame([{
                'symbol': ticker, 
                'close': df['close'].iloc[-1], # Use last known close as placeholder for today's 'prev_close' for vol calc
                'volume': current_session_volume,
                'date': dt.date.today().isoformat()
            }])
            df = pd.concat([df, df_today], ignore_index=True)
            
            # Calculate and return the structured features
            features = calculate_risk_features(df, current_session_volume)

            if features is None:
                return f"Error: Could not calculate risk features for {ticker}. Need at least 21 days of data."
            
            return features

        except requests.exceptions.RequestException as e:
            return f"API Request Error for {ticker}: {e}"
        except Exception as e:
            # Catching general errors, e.g., in pandas calculation
            return f"An unexpected error occurred while calculating features for {ticker}: {e}"