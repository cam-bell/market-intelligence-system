#!/usr/bin/env python
import sys
import warnings
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from .crew import MarketSentiment

# Load environment variables from the repo root.
# Path from: src/market_sentiment/main.py -> repo-root/.env
env_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path=env_path)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the market sentiment analysis crew.
    """
    # Default inputs - can be overridden via command line or environment
    inputs = {
        'market_focus': os.getenv('MARKET_FOCUS', 'AI and technology stocks'),
        'tickers': os.getenv('TICKERS', 'AAPL').split(','),
        'asset_list': os.getenv('ASSET_LIST', 'AAPL').split(','),
        'metrics': os.getenv('METRICS', 'price,volume,volatility').split(','),
        'platforms': os.getenv('PLATFORMS', 'reddit,twitter,stocktwits').split(','),
        'time_window': os.getenv('TIME_WINDOW', '24'),
        'current_date': str(datetime.now())
    }

    # Override with command line args if provided
    if len(sys.argv) > 1:
        # First arg can be tickers: python main.py AAPL,MSFT,NVDA
        if sys.argv[1]:
            inputs['tickers'] = sys.argv[1].split(',')
            inputs['asset_list'] = sys.argv[1].split(',')

    # Create and run the crew
    result = MarketSentiment().crew().kickoff(inputs=inputs)

    # Print the result
    print("\n\n=== MARKET SENTIMENT ANALYSIS ===\n\n")
    print(result.raw)
    
    return result


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'market_focus': os.getenv('MARKET_FOCUS', 'AI and technology stocks'),
        'tickers': os.getenv('TICKERS', 'AAPL,MSFT,NVDA,GOOGL,TSLA').split(','),
        'asset_list': os.getenv('ASSET_LIST', 'AAPL,MSFT,NVDA,GOOGL,TSLA').split(','),
        'metrics': os.getenv('METRICS', 'price,volume,volatility').split(','),
        'platforms': os.getenv('PLATFORMS', 'reddit,twitter,stocktwits').split(','),
        'time_window': os.getenv('TIME_WINDOW', '24'),
        'current_date': str(datetime.now())
    }
    try:
        MarketSentiment().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        MarketSentiment().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'market_focus': os.getenv('MARKET_FOCUS', 'AI and technology stocks'),
        'tickers': os.getenv('TICKERS', 'AAPL,MSFT,NVDA,GOOGL,TSLA').split(','),
        'asset_list': os.getenv('ASSET_LIST', 'AAPL,MSFT,NVDA,GOOGL,TSLA').split(','),
        'metrics': os.getenv('METRICS', 'price,volume,volatility').split(','),
        'platforms': os.getenv('PLATFORMS', 'reddit,twitter,stocktwits').split(','),
        'time_window': os.getenv('TIME_WINDOW', '24'),
        'current_date': str(datetime.now())
    }

    try:
        MarketSentiment().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception(
            "No trigger payload provided. Please provide JSON payload as argument."
        )

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        'market_focus': '',
        'tickers': [],
        'asset_list': [],
        'metrics': [],
        'platforms': [],
        'time_window': '24',
        'current_date': str(datetime.now())
    }

    try:
        result = MarketSentiment().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")


if __name__ == "__main__":
    run()
