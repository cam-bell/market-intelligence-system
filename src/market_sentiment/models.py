# src/schemas/market_data.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from uuid import UUID, uuid4


class NewsArticle(BaseModel):
    source: str
    title: str
    url: str
    snippet: str = Field(default="", max_length=500)
    timestamp: datetime
    ticker_mentions: List[str] = []
    toxicity_score: float = Field(default=0.0, ge=0, le=1)
    # NEW: Add sentiment fields if available
    sentiment_score: Optional[float] = Field(default=None, ge=-1, le=1)
    sentiment_reasoning: Optional[str] = Field(default=None, max_length=500)

class NewsCorpus(BaseModel):
    articles: List[NewsArticle] = Field(default_factory=list)
    fetch_timestamp: datetime = Field(default_factory=datetime.now)
    source: str = "FinancialNewsTool"

class SocialPost(BaseModel):
    platform: str
    post_id: str
    snippet: str = Field(max_length=300)
    timestamp: datetime
    ticker_mentions: List[str] = []
    engagement_score: float = Field(default=0.0, ge=0, le=1)

class SocialCorpus(BaseModel):
    posts: List[SocialPost] = Field(default_factory=list)
    fetch_timestamp: datetime = Field(default_factory=datetime.now)
    platforms: List[str] = Field(default_factory=list)

class SentimentProfile(BaseModel):
    ticker: str
    timestamp: datetime = Field(default_factory=datetime.now)
    news_sentiment_raw: float = Field(ge=-5, le=5)
    news_sentiment_normalized: float = Field(ge=-1, le=1)
    social_sentiment_raw: float = Field(ge=-5, le=5)
    social_sentiment_normalized: float = Field(ge=-1, le=1)
    blended_sentiment: float = Field(ge=-1, le=1)
    sentiment_confidence: float = Field(ge=0, le=1)
    key_themes: List[str] = Field(default_factory=list, max_length=5)

# Update existing RiskAssessment to match proposed spec:
class RiskAssessment(BaseModel):
    ticker: str  # Add this
    timestamp: datetime = Field(default_factory=datetime.now)
    risk_score: float = Field(ge=0, le=100)  # Change from int to float
    risk_level: str = Field(pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    component_scores: Dict[str, float] = Field(default_factory=dict)  # Add this
    anomaly_flags: List[str] = Field(default_factory=list)  # Add this
    risk_rationale: str = Field(max_length=500)  # Add this
    # Remove: confidence_interval, var_calculation, volatility_index, anomaly_detected

# Add Signal model:
class Signal(BaseModel):
    ticker: str
    risk_score: float
    risk_level: str
    sentiment_summary: str
    volatility_summary: str
    key_drivers: List[str]
    recommended_actions: List[str]

# Update MarketSignals:
class MarketSignals(BaseModel):
    run_id: UUID = Field(default_factory=uuid4)  # Add this
    timestamp: datetime = Field(default_factory=datetime.now)
    signals: List[Signal]  # Change from flat structure
    metadata: Dict = Field(default_factory=dict)
    # Remove: assets, net_sentiment, risk_score, themes, market_data, sentiment_details, risk_details, alert_triggered