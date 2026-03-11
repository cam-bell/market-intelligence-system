import os
import requests
import datetime as dt
import numpy as np
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from datetime import datetime
from typing import Optional

# --- Pydantic Output Model for the Tool ---
class MarketSnapshot(BaseModel):
    """Real-time data for a single asset (the A-Model)."""
    symbol: str = Field(description="The ticker symbol of the asset.")
    exchange: str = Field(description="The exchange the asset trades on (e.g., XNAS).")
    timestamp_utc: dt.datetime = Field(description="The timestamp of the snapshot, standardized to UTC.")
    last_price: float = Field(description="The last traded price of the asset.")
    pct_change_1d: float = Field(description="Percentage change (e.g., 0.015 for +1.5%) vs previous close.")
    volume: int = Field(description="Current session volume (intraday).")
    prev_close: Optional[float] = Field(description="The closing price from the previous trading day.")
    day_high: Optional[float] = Field(description="Highest price reached during the current trading day.")
    day_low: Optional[float] = Field(description="Lowest price reached during the current trading day.")

# --- The CrewAI Tool Class ---
class FinancialDataTool(BaseTool):
    name: str = "Financial Data Tool"
    description: str = (
        "Fetches the current real-time market snapshot for a given stock ticker, "
        "including last price, volume, and daily high/lows. Required for immediate risk assessment."
    )

    def _run(self, ticker: str) -> MarketSnapshot:
        """
        Fetches the current market snapshot data from the Marketstack API.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL').

        Returns:
            MarketSnapshot: The structured real-time market data.
        """
        ACCESS_KEY = os.environ.get("MARKETSTACK_API_KEY")
        BASE_URL = "https://api.marketstack.com/v2/eod"
        
        params = {
            'access_key': ACCESS_KEY,
            'symbols': ticker,
        }
        
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json().get('data', [])
            
            if not data:
                return f"Error: No data returned for ticker {ticker}. Check symbol validity."

            snapshot = data[0]

            # Get previous day's close if available
            prev_close_value = None
            if len(data) > 1:
                prev_close_value = data[1].get('close')
            
            # Helper to safely convert to float/int, defaulting to None/0 if missing
            def safe_cast(value, default_type):
                if value is None:
                    return None if default_type == float else 0
                try:
                    return default_type(value)
                except (ValueError, TypeError):
                    return None if default_type == float else 0

            # Calculate pct_change_1d
            prev_close = safe_cast(prev_close_value, float)
            last_price = safe_cast(snapshot.get('close'), float)
            pct_change_1d = 0.0
            if prev_close and last_price and prev_close != 0:
                pct_change_1d = (last_price - prev_close) / prev_close
            
            
            return MarketSnapshot(
                symbol=snapshot.get('symbol'),
                exchange=snapshot.get('exchange'),
                timestamp_utc=dt.datetime.fromisoformat(snapshot.get('date').replace('Z', '+00:00')),
                last_price=last_price,
                pct_change_1d=pct_change_1d,
                volume=safe_cast(snapshot.get('volume'), int),
                prev_close=prev_close,
                day_high=safe_cast(snapshot.get('high'), float),
                day_low=safe_cast(snapshot.get('low'), float),
            )

        except requests.exceptions.RequestException as e:
            return f"API Request Error for {ticker}: {e}"
        except Exception as e:
            return f"An unexpected error occurred while processing data for {ticker}: {e}"