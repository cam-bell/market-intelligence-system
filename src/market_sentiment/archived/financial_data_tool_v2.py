import os
import requests
import datetime as dt
from typing import Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

# --- Pydantic Output Model for the Tool ---
class MarketSnapshotV2(BaseModel):
    """Real-time data for a single asset (the A-Model)."""
    symbol: str = Field(description="The ticker symbol of the asset.")
    timestamp_utc: dt.datetime = Field(description="The timestamp of the snapshot, standardized to UTC.")
    last_price: float = Field(description="The last traded price of the asset.")
    prev_close: Optional[float] = Field(description="The closing price from the previous trading day.")
    volume: int = Field(description="Current session volume (intraday).")

# --- The CrewAI Tool Class ---
class FinancialDataToolV2(BaseTool):
    name: str = "Financial Data Tool V2"
    description: str = (
        "Fetches the current real-time market snapshot for a given stock ticker, "
        "including last price and current volume from the Marketstack Real-Time endpoint. "
        "Goal: pricing, volume + volatility data."
    )

    def _run(self, ticker: str) -> MarketSnapshotV2:
        """
        Fetches the current market snapshot data from the Marketstack API.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL').

        Returns:
            MarketSnapshotV2 | str: The structured real-time market data or an error string.
        """
        ACCESS_KEY = os.environ.get("MARKETSTACK_API_KEY")
        # Using the End of Day (EOD) endpoint - returns historical data sorted by date DESC
        BASE_URL = "https://api.marketstack.com/v2/eod" 
        
        params = {
            'access_key': ACCESS_KEY,
            'symbols': ticker,
            'limit': 2  # Get 2 most recent days: [0] = latest, [1] = previous day
        }
        
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json().get('data', [])
            
            if not data:
                return f"Error: No data returned for ticker {ticker}. Check symbol validity."

            # Data is already sorted DESC by date (newest first)
            # Get the most recent day's data
            latest_snapshot = data[0]
            
            # Get previous day's close if available (data[1])
            prev_close_value = None
            if len(data) > 1:
                prev_close_value = data[1].get('close')

            # Helper to safely cast
            def safe_float(value):
                return float(value) if value is not None else None
            def safe_int(value):
                return int(value) if value is not None else 0

            return MarketSnapshotV2(
                symbol=latest_snapshot.get('symbol'),
                timestamp_utc=dt.datetime.fromisoformat(latest_snapshot.get('date').replace('Z', '+00:00')),
                last_price=safe_float(latest_snapshot.get('close')),  # Use 'close' instead of 'last'
                prev_close=safe_float(prev_close_value),  # Get from data[1] if available
                volume=safe_int(latest_snapshot.get('volume')),
            )

        except requests.exceptions.RequestException as e:
            return f"API Request Error for {ticker}: {e}"
        except Exception as e:
            return f"An unexpected error occurred while processing real-time data for {ticker}: {e}"