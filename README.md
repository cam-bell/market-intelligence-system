# Market Intelligence System

Multi-agent market intelligence and risk analysis workflow built with CrewAI.

This project ingests financial news, market data, and social sentiment, then synthesizes those signals into structured risk assessments and final market signal outputs. The repository preserves the original project layout from the last known runnable `market_sentiment` state so it can be moved into GitHub with minimal path or import drift.

## What It Does

- Aggregates financial news from multiple sources.
- Pulls real-time and historical market data for selected tickers.
- Collects alternative sentiment signals from social platforms.
- Uses financial NLP to score sentiment and extract themes.
- Computes a unified risk score from sentiment, volatility, and anomaly features.
- Compiles structured output as typed market signal objects.

## Agent Workflow

The current workflow is defined in [`src/market_sentiment/config/agents.yaml`](/Users/cameronbell/Projects/market-intelligence-system/src/market_sentiment/config/agents.yaml) and [`src/market_sentiment/config/tasks.yaml`](/Users/cameronbell/Projects/market-intelligence-system/src/market_sentiment/config/tasks.yaml).

### Agents

- `data_scout`: ingests and filters market-moving news.
- `market_data_fetcher`: retrieves real-time and historical market metrics.
- `alt_data_scout`: captures retail and social sentiment.
- `sentiment_analyst`: converts raw text into normalized sentiment and themes.
- `risk_engine`: computes a unified risk score.
- `signal_synthesizer`: validates and compiles final structured output.

### Pipeline

1. Ingest news data.
2. Fetch market metrics.
3. Ingest social sentiment.
4. Synthesize sentiment from news and social context.
5. Calculate risk score from sentiment and market features.
6. Compile final `MarketSignals` output.

## Repository Layout

```text
.
├── knowledge/                  # Glossary, policies, calibration, user preferences
├── src/market_sentiment/
│   ├── archived/               # Legacy experiments and notebooks
│   ├── config/                 # Agent and task definitions
│   ├── memory/                 # Runtime memory artifacts
│   ├── output/                 # Generated output artifacts
│   ├── tools/                  # Custom CrewAI tools
│   ├── crew.py                 # Crew and agent wiring
│   ├── main.py                 # Entrypoints for run/train/replay/test
│   └── models.py               # Pydantic models for pipeline outputs
├── tests/                      # Present but currently empty
├── .env.example                # Sanitized environment template
├── pyproject.toml              # Main project metadata and dependencies
└── uv.lock                     # Locked dependency graph
```

## Core Models

The pipeline uses structured Pydantic models defined in [`src/market_sentiment/models.py`](/Users/cameronbell/Projects/market-intelligence-system/src/market_sentiment/models.py), including:

- `NewsCorpus`
- `SocialCorpus`
- `SentimentProfile`
- `RiskAssessment`
- `MarketSignals`

These models make the system easier to reason about, validate, and diagram.

## Setup

### Prerequisites

- Python `>=3.10,<3.14`
- `uv` recommended for environment and dependency management

### Install

```bash
uv sync
```

If you prefer the CrewAI install flow:

```bash
crewai install
```

### Environment

Copy the example file and fill in the keys you actually use:

```bash
cp .env.example .env
```

Expected environment variables:

- `OPENAI_API_KEY`
- `SERPER_API_KEY`
- `MARKETSTACK_API_KEY`
- `GNEWS_API_KEY`
- `MASSIVE_API_KEY`
- `ALPHAVANTAGE_API_KEY`
- `FINNHUB_API_KEY`
- `MARKET_FOCUS`
- `TICKERS`
- `ASSET_LIST`
- `METRICS`
- `PLATFORMS`
- `TIME_WINDOW`

Note: the code currently loads `.env` from the repository root.

## Running

Run the crew:

```bash
uv run market_sentiment
```

Or use the script entrypoints from [`pyproject.toml`](/Users/cameronbell/Projects/market-intelligence-system/pyproject.toml):

```bash
uv run run_crew
uv run train -- <iterations> <filename>
uv run replay -- <task_id>
uv run test -- <iterations> <eval_llm>
```

You can also pass tickers directly to the main module:

```bash
uv run python -m market_sentiment.main AAPL,MSFT,NVDA
```

## Outputs

The current task configuration writes intermediate and final artifacts under [`src/market_sentiment/output`](/Users/cameronbell/Projects/market-intelligence-system/src/market_sentiment/output), including:

- `news_corpus.json`
- `market_snapshots.json`
- `social_corpus.json`
- `sentiment_profiles.json`
- `risk_assessments.json`
- `market_signals.json`

## Key Files For Diagrams

If you are creating system architecture, data flow, service, or agent workflow diagrams, start with:

- [`src/market_sentiment/crew.py`](/Users/cameronbell/Projects/market-intelligence-system/src/market_sentiment/crew.py)
- [`src/market_sentiment/config/agents.yaml`](/Users/cameronbell/Projects/market-intelligence-system/src/market_sentiment/config/agents.yaml)
- [`src/market_sentiment/config/tasks.yaml`](/Users/cameronbell/Projects/market-intelligence-system/src/market_sentiment/config/tasks.yaml)
- [`src/market_sentiment/models.py`](/Users/cameronbell/Projects/market-intelligence-system/src/market_sentiment/models.py)
- [`src/market_sentiment/tools`](/Users/cameronbell/Projects/market-intelligence-system/src/market_sentiment/tools)
- [`knowledge`](/Users/cameronbell/Projects/market-intelligence-system/knowledge)

## Current State

This repository reflects a strong prototype rather than a fully hardened production service.

- The layout is intentionally preserved from the original runnable project.
- Some runtime artifacts are still checked in under `src/market_sentiment/memory` and `src/market_sentiment/output`.
- `src/market_sentiment/archived` contains legacy experiments.
- `tests/` is currently empty.
- The root README has been modernized, but the codebase itself has not been refactored in this pass.

Recommended cleanup notes are documented in [`RECOMMENDED_ADJUSTMENTS.md`](/Users/cameronbell/Projects/market-intelligence-system/RECOMMENDED_ADJUSTMENTS.md).
