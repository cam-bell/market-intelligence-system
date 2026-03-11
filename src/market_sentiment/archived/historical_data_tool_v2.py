import os
import requests
import datetime as dt
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from typing import Optional

# --- Pydantic Output Model for the Tool ---
class RiskFeatureSetV2(BaseModel):
    """Derived risk features (the B-Model) calculated from historical OHLCV data."""
    symbol: str = Field(description="The ticker symbol of the asset.")
    realized_vol_21d_ann: float = Field(
        description="Annualized Realized Volatility over the last 21 trading days (252-day convention)."
    )
    volume_zscore_20d: float = Field(
        description="The Z-score of today's volume compared to the 20-day average volume."
    )
    max_drawdown_21d: Optional[float] = Field(
        description="The largest peak-to-trough decline over the last 21 days."
    )
    
# --- The Quant Calculation Functions ---

def calculate_volatility_and_drawdown(df: pd.DataFrame) -> tuple[float, Optional[float]]:
    """Calculates realized volatility and max drawdown from the DataFrame."""
    if len(df) < 21:
        return 0.0, 0.0
    
    # Realized Volatility (21-day, Annualized)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    DAILY_TRADING_DAYS = 252
    VOL_WINDOW = 21
    
    realized_vol_21d = df['log_return'].iloc[-VOL_WINDOW:].std()
    realized_vol_21d_ann = realized_vol_21d * np.sqrt(DAILY_TRADING_DAYS)

    # Max Drawdown (21-day)
    window_prices = df['close'].iloc[-VOL_WINDOW:]
    peak = window_prices.cummax()
    drawdown = (window_prices / peak) - 1.0
    max_drawdown_21d = drawdown.min()
    
    return realized_vol_21d_ann, max_drawdown_21d

def calculate_volume_zscore(df: pd.DataFrame, current_volume: int) -> float:
    """Calculates the Z-score of the current volume relative to historical volume."""
    VOL_Z_WINDOW = 20
    if len(df) < VOL_Z_WINDOW:
        return 0.0
        
    avg_volume = df['volume'].iloc[-VOL_Z_WINDOW:].mean()
    std_volume = df['volume'].iloc[-VOL_Z_WINDOW:].std()
    
    volume_zscore_20d = (current_volume - avg_volume) / std_volume if std_volume > 0 else 0.0
    return volume_zscore_20d

# --- The CrewAI Tool Class ---
class HistoricalDataToolV2(BaseTool):
    name: str = "Historical Data Tool V2"
    description: str = (
        "Fetches historical OHLCV data (up to 40 days) from the EOD endpoint to compute "
        "derived risk features like annualized realized volatility and volume Z-score. "
        "Goal: (OHLCV, volatility window)."
    )

    def _run(self, ticker: str, current_session_volume: int) -> RiskFeatureSetV2 | str:
        """
        Fetches EOD data and calculates derived features.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL').
            current_session_volume (int): The current day's volume from the MarketSnapshot (V2).

        Returns:
            RiskFeatureSetV2 | str: The structured derived risk metrics or an error string.
        """
        ACCESS_KEY = os.environ.get("MARKETSTACK_ACCESS_KEY", "YOUR_API_KEY_HERE")
        # Using the End-of-Day endpoint (v2/eod)
        BASE_URL = "https://api.marketstack.com/v2/eod"
        
        # Request a buffer of data to ensure we have 21 trading days
        date_from = (dt.date.today() - dt.timedelta(days=40)).strftime('%Y-%m-%d')
        
        params = {
            'access_key': ACCESS_KEY,
            'symbols': ticker,
            'date_from': date_from,
            'limit': 40,
            'sort': 'ASC'
        }
        
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data_list = response.json().get('data', [])
            
            if not data_list:
                return f"Error: Not enough historical EOD data returned for {ticker}."

            df = pd.DataFrame(data_list).sort_values(by='date', ascending=True)
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df = df.dropna(subset=['close', 'volume'])
            
            if len(df) < 21:
                return f"Error: Need at least 21 days of data to calculate full risk features for {ticker}."
            
            # 1. Calculate volatility and drawdown
            realized_vol_21d_ann, max_drawdown_21d = calculate_volatility_and_drawdown(df)
            
            # 2. Calculate volume Z-score (uses the current session volume provided as an argument)
            volume_zscore_20d = calculate_volume_zscore(df, current_session_volume)

            return RiskFeatureSetV2(
                symbol=ticker,
                realized_vol_21d_ann=realized_vol_21d_ann,
                volume_zscore_20d=volume_zscore_20d,
                max_drawdown_21d=max_drawdown_21d,
            )

        except requests.exceptions.RequestException as e:
            return f"API Request Error for {ticker}: {e}"
        except Exception as e:
            return f"An unexpected error occurred during feature calculation for {ticker}: {e}"