"""Enhanced Risk Calculation Tool - Best of All Approaches."""
from pydantic import BaseModel, Field, field_validator
from crewai.tools import BaseTool
from datetime import datetime

from ..models import RiskAssessment, SentimentProfile
from .historical_data_tool import RiskFeatureSet


class RiskConfig(BaseModel):
    """Configurable risk calculation parameters."""
    # Weights (must sum to 1.0)
    ALPHA: float = 0.40  # Sentiment Weight (S)
    BETA: float = 0.30   # Volatility Weight (V)
    GAMMA: float = 0.20  # Anomaly Weight (A)
    DELTA: float = 0.10  # Macro Weight (M)

    # Risk level thresholds (0-100 scale)
    CRITICAL_THRESHOLD: float = 80.0
    HIGH_THRESHOLD: float = 60.0
    MEDIUM_THRESHOLD: float = 30.0

    # Anomaly detection thresholds
    VOL_SPIKE_THRESHOLD: float = 0.35
    ZSCORE_THRESHOLD: float = 2.5
    EXTREME_SENTIMENT_THRESHOLD: float = 0.7
    LOW_CONFIDENCE_THRESHOLD: float = 0.4

    # Component scaling (for transparency)
    VOL_MAX_THRESHOLD: float = 0.50  # 50% vol = max risk
    ZSCORE_MAX: float = 3.0  # Z=3.0 = max risk


def calculate_unified_risk(
    ticker: str,
    sentiment_profile: SentimentProfile,
    risk_features: RiskFeatureSet,
    macro_risk: float = 0.0,
    config: RiskConfig = RiskConfig()
) -> RiskAssessment:
    """
    Calculates unified risk: R = α*S + β*V + γ*A + δ*M.
    
    Uses structured inputs from context tasks for type safety.
    """
    # Validate weights sum to 1.0
    total_weight = config.ALPHA + config.BETA + config.GAMMA + config.DELTA
    if abs(total_weight - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    # Component calculations (0-100 scale each)
    net_sentiment = sentiment_profile.blended_sentiment
    S = max(0, -net_sentiment) * 100  # Invert: bearish = risk

    realized_vol = risk_features.realized_vol_21d_ann
    V = min(realized_vol / config.VOL_MAX_THRESHOLD * 100, 100)

    volume_zscore = risk_features.volume_zscore_20d
    A = min(max(0, volume_zscore) / config.ZSCORE_MAX * 100, 100)

    M = macro_risk * 100

    # Weighted formula
    risk_score = (
        config.ALPHA * S +
        config.BETA * V +
        config.GAMMA * A +
        config.DELTA * M
    )
    risk_score = max(0.0, min(100.0, risk_score))

    # Component scores
    component_scores = {
        'sentiment_risk': round(S, 2),
        'volatility_risk': round(V, 2),
        'volume_anomaly_risk': round(A, 2),
        'macro_risk': round(M, 2)
    }

    # Anomaly detection
    anomaly_flags = []
    if realized_vol > config.VOL_SPIKE_THRESHOLD:
        anomaly_flags.append("UNUSUAL_VOLATILITY")
    if volume_zscore > config.ZSCORE_THRESHOLD:
        anomaly_flags.append("UNUSUAL_VOLUME")
    if abs(net_sentiment) > config.EXTREME_SENTIMENT_THRESHOLD:
        anomaly_flags.append("EXTREME_SENTIMENT")
    if sentiment_profile.sentiment_confidence < config.LOW_CONFIDENCE_THRESHOLD:
        anomaly_flags.append("LOW_SENTIMENT_CONFIDENCE")

    # Risk level
    if risk_score >= config.CRITICAL_THRESHOLD:
        risk_level = "CRITICAL"
    elif risk_score >= config.HIGH_THRESHOLD:
        risk_level = "HIGH"
    elif risk_score >= config.MEDIUM_THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Generate rationale (enhanced from Option 3's simplicity)
    rationale_parts = []
    if S > 20:
        rationale_parts.append(f"negative sentiment ({S:.0f} pts)")
    if V > 15:
        rationale_parts.append(f"elevated volatility ({V:.0f} pts)")
    if A > 10:
        rationale_parts.append(f"abnormal volume ({A:.0f} pts)")
    if M > 5:
        rationale_parts.append(f"macro risk ({M:.0f} pts)")
    if anomaly_flags:
        rationale_parts.append(f"anomalies: {', '.join(anomaly_flags)}")

    risk_rationale = (
        f"Risk driven by: {'; '.join(rationale_parts)}"
        if rationale_parts
        else "Risk within normal parameters"
    )

    return RiskAssessment(
        ticker=ticker,
        timestamp=datetime.now(),
        risk_score=round(risk_score, 2),
        risk_level=risk_level,
        component_scores=component_scores,
        anomaly_flags=anomaly_flags,
        risk_rationale=risk_rationale[:500]
    )


class RiskCalculationToolInput(BaseModel):
    """Input schema for Risk Calculation Tool."""
    ticker: str = Field(..., description="Ticker symbol to assess")
    sentiment_profile: SentimentProfile = Field(
        ..., description="Sentiment analysis from synthesize_sentiment task"
    )
    risk_features: RiskFeatureSet = Field(
        ..., description="Risk features from fetch_market_metrics task"
    )
    macro_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Optional macro risk factor (0.0 to 1.0)"
    )

    @field_validator('sentiment_profile', 'risk_features', mode='before')
    @classmethod
    def convert_dict_to_model(cls, v, info):
        if isinstance(v, dict):
            if info.field_name == 'sentiment_profile':
                return SentimentProfile(**v)
            elif info.field_name == 'risk_features':
                return RiskFeatureSet(**v)
        return v


class RiskCalculationTool(BaseTool):
    """Quantitative risk calculation using VaR methodology."""

    name: str = "Risk Calculation Tool"
    description: str = (
        "Calculates unified risk score (0-100 scale) using formula: "
        "R = α*S + β*V + γ*A + δ*M where α=0.40, β=0.30, γ=0.20, δ=0.10. "
        "Takes SentimentProfile and RiskFeatureSet from context tasks. "
        "Returns RiskAssessment with component scores, anomaly flags, and "
        "rationale. Uses strict mathematical calculations - NEVER delegates to LLM."
    )
    args_schema: type[BaseModel] = RiskCalculationToolInput

    def _run(
        self,
        ticker: str,
        sentiment_profile: SentimentProfile,
        risk_features: RiskFeatureSet,
        macro_risk: float = 0.0
    ) -> RiskAssessment | str:
        """Calculate unified risk score using quantitative methods."""
        try:
            return calculate_unified_risk(
                ticker, sentiment_profile, risk_features, macro_risk
            )
        except Exception as e:
            return f"Error executing risk calculation: {e}"