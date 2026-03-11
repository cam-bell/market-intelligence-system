"""Sentiment Analysis Tool using FinBERT for financial text."""
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator
import numpy as np
from loguru import logger

from ..models import NewsCorpus, SocialCorpus, SentimentProfile


class SentimentAnalysisToolInput(BaseModel):
    """Input schema for Sentiment Analysis Tool."""
    news_corpus: NewsCorpus = Field(..., description="News corpus from ingestion task")
    social_corpus: SocialCorpus = Field(..., description="Social media corpus from ingestion task")
    ticker: str = Field(..., description="Ticker symbol to analyze")

    @field_validator('news_corpus', 'social_corpus', mode='before')
    @classmethod
    def convert_dict_to_model(cls, v):
        if isinstance(v, dict):
            if 'articles' in v:
                # Ensure all articles have snippet field and parse timestamps
                articles = v.get('articles', [])
                for article in articles:
                    if 'snippet' not in article or not article.get('snippet'):
                        article['snippet'] = article.get('title', '')
                    # Parse timestamp if it's a string
                    if 'timestamp' in article and isinstance(article['timestamp'], str):
                        from datetime import datetime
                        try:
                            # Handle ISO format with or without Z
                            ts_str = article['timestamp'].replace('Z', '+00:00')
                            article['timestamp'] = datetime.fromisoformat(ts_str)
                        except (ValueError, AttributeError):
                            article['timestamp'] = datetime.now()
                return NewsCorpus(**v)
            elif 'posts' in v:  # Fixed: removed 'or not v' condition
                # Ensure fetch_timestamp is set and parsed
                fetch_timestamp = v.get('fetch_timestamp')
                if fetch_timestamp is None:
                    from datetime import datetime
                    fetch_timestamp = datetime.now()
                elif isinstance(fetch_timestamp, str):
                    from datetime import datetime
                    try:
                        # Handle ISO format with or without timezone
                        ts_str = fetch_timestamp.replace('Z', '+00:00') if 'Z' in fetch_timestamp else fetch_timestamp
                        fetch_timestamp = datetime.fromisoformat(ts_str)
                    except (ValueError, AttributeError):
                        fetch_timestamp = datetime.now()
                return SocialCorpus(
                    posts=v.get('posts', []), 
                    fetch_timestamp=fetch_timestamp,
                    platforms=v.get('platforms', [])
                )
        return v

class SentimentAnalysisTool(BaseTool):
    """Financial sentiment analysis using FinBERT transformer model."""
    
    name: str = "Sentiment Analysis Tool"
    description: str = (
        "Analyzes financial text sentiment using FinBERT model. "
        "Takes news and social corpora and returns normalized sentiment scores (-1.0 to +1.0) "
        "along with key themes. Uses specialized financial NLP model for accurate scoring."
    )
    args_schema: type[BaseModel] = SentimentAnalysisToolInput

    def __init__(self):
        super().__init__()
        self._finbert = None
        self._llm = None

    @property
    def finbert(self):
        """Lazy load FinBERT pipeline."""
        if self._finbert is None:
            try:
                from transformers import pipeline
                import torch
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._finbert = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device=device if device == "cuda" else -1
                )
                logger.info(f"FinBERT loaded on {device}")
            except ImportError:
                logger.error("transformers or torch not installed. Install with: pip install transformers torch")
                raise
        return self._finbert

    @property
    def llm(self):
        """Lazy load LLM for theme extraction."""
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.2,
                    max_tokens=500
                )
                logger.info("LLM loaded for theme extraction")
            except ImportError:
                logger.error("langchain_openai not installed")
                raise
        return self._llm

    def _run(
        self,
        news_corpus: NewsCorpus | dict,
        social_corpus: SocialCorpus | dict,
        ticker: str
    ) -> SentimentProfile:
        """
        Analyze sentiment from news and social corpora.
        
        Process:
        1. Score each text snippet with FinBERT
        2. Aggregate raw sentiment scores
        3. Normalize scores (-1.0 to +1.0)
        4. Extract themes using LLM
        5. Blend news and social sentiment
        """
    
        try:
            # Convert dicts to models if needed
            if isinstance(news_corpus, dict):
                # Ensure all articles have snippet field and parse timestamps
                articles = news_corpus.get('articles', [])
                for article in articles:
                    if 'snippet' not in article or not article.get('snippet'):
                        article['snippet'] = article.get('title', '')
                    # Parse timestamp if it's a string
                    if 'timestamp' in article and isinstance(article['timestamp'], str):
                        from datetime import datetime
                        try:
                            ts_str = article['timestamp'].replace('Z', '+00:00')
                            article['timestamp'] = datetime.fromisoformat(ts_str)
                        except (ValueError, AttributeError):
                            article['timestamp'] = datetime.now()
                news_corpus = NewsCorpus(**news_corpus)
            
            if isinstance(social_corpus, dict):
                fetch_timestamp = social_corpus.get('fetch_timestamp')
                if fetch_timestamp is None:
                    from datetime import datetime
                    fetch_timestamp = datetime.now()
                elif isinstance(fetch_timestamp, str):
                    from datetime import datetime
                    try:
                        ts_str = fetch_timestamp.replace('Z', '+00:00') if 'Z' in fetch_timestamp else fetch_timestamp
                        fetch_timestamp = datetime.fromisoformat(ts_str)
                    except (ValueError, AttributeError):
                        fetch_timestamp = datetime.now()
                social_corpus = SocialCorpus(
                    posts=social_corpus.get('posts', []),
                    fetch_timestamp=fetch_timestamp,
                    platforms=social_corpus.get('platforms', [])
                )
            
            # Step 1: Score news articles with FinBERT
            news_scores = []
            for article in news_corpus.articles:
                score = self._score_text(article.snippet or article.title)
                news_scores.append(score)
            
            # Step 2: Score social posts with FinBERT
            social_scores = []
            for post in social_corpus.posts:
                score = self._score_text(post.snippet)
                social_scores.append(score)
            
            # Step 3: Aggregate raw scores
            news_sentiment_raw = self._aggregate_scores(news_scores) if news_scores else 0.0
            social_sentiment_raw = self._aggregate_scores(social_scores) if social_scores else 0.0
            
            # Step 4: Normalize scores (-1.0 to +1.0)
            # Simple normalization: map FinBERT scores to -1 to +1 range
            news_sentiment_normalized = self._normalize_score(news_sentiment_raw)
            social_sentiment_normalized = self._normalize_score(social_sentiment_raw)
            
            # Step 5: Extract themes using LLM
            themes = self._extract_themes(news_corpus, social_corpus)
            
            # Step 6: Blend scores (60% news, 40% social per common practice)
            blended_sentiment = 0.6 * news_sentiment_normalized + 0.4 * social_sentiment_normalized
            
            # Step 7: Calculate confidence from score consistency
            sentiment_confidence = self._calculate_confidence(news_scores, social_scores)
            
            return SentimentProfile(
                ticker=ticker,
                news_sentiment_raw=round(news_sentiment_raw, 3),
                news_sentiment_normalized=round(news_sentiment_normalized, 3),
                social_sentiment_raw=round(social_sentiment_raw, 3),
                social_sentiment_normalized=round(social_sentiment_normalized, 3),
                blended_sentiment=round(blended_sentiment, 3),
                sentiment_confidence=round(sentiment_confidence, 3),
                key_themes=themes[:5]  # Limit to 5 themes
            )
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise

    def _score_text(self, text: str) -> dict:
        """Run FinBERT sentiment analysis on text."""
        if not text:
            return {"label": "neutral", "score": 0.5}
        
        # FinBERT max length is 512 tokens
        text_truncated = text[:512]
        
        try:
            finbert_pipeline = self.finbert
            result = finbert_pipeline(text_truncated)[0]
            return result
        except (ImportError, AttributeError, IndexError) as e:
            logger.warning(f"FinBERT scoring failed: {e}")
            return {"label": "neutral", "score": 0.5}

    def _aggregate_scores(self, scores: list[dict]) -> float:
        """Aggregate FinBERT scores into a single value."""
        if not scores:
            return 0.0
        
        # Convert FinBERT labels to numeric values
        # positive = +1, negative = -1, neutral = 0
        numeric_scores = []
        for score_dict in scores:
            label = score_dict.get("label", "neutral").lower()
            confidence = score_dict.get("score", 0.5)
            
            if label == "positive":
                numeric_scores.append(confidence)  # +0.5 to +1.0
            elif label == "negative":
                numeric_scores.append(-confidence)  # -0.5 to -1.0
            else:  # neutral
                numeric_scores.append(0.0)
        
        return np.mean(numeric_scores) if numeric_scores else 0.0

    def _normalize_score(self, raw_score: float) -> float:
        """Normalize raw score to -1.0 to +1.0 range."""
        # Clamp to valid range
        return max(-1.0, min(1.0, raw_score))

    def _extract_themes(self, news_corpus: NewsCorpus, social_corpus: SocialCorpus) -> list[str]:
        """Extract key themes using LLM."""
        try:
            # Prepare sample texts for theme extraction
            news_samples = [a.title for a in news_corpus.articles[:10]]
            social_samples = [p.snippet[:100] for p in social_corpus.posts[:20]]
            
            prompt = f"""Extract 3-5 key financial themes from the following market commentary.

News Headlines:
{chr(10).join(f"- {title}" for title in news_samples[:10])}

Social Media Posts:
{chr(10).join(f"- {post}" for post in social_samples[:20])}

Return ONLY a JSON array of theme strings, e.g. ["AI Regulation Concerns", "Earnings Beat Expectations"].
Do not include any explanation, only the JSON array."""

            llm_client = self.llm
            response = llm_client.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON array from response
            import json
            import re
            
            # Try to find JSON array in response
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                themes = json.loads(json_match.group(0))
                return themes[:5] if isinstance(themes, list) else []
            
            logger.warning("Could not parse themes from LLM response")
            return []
        except (ImportError, AttributeError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Theme extraction failed: {e}")
            return []

    def _calculate_confidence(self, news_scores: list, social_scores: list) -> float:
        """Calculate confidence from score consistency."""
        all_scores = news_scores + social_scores
        if not all_scores:
            return 0.0
        
        # Extract numeric values for variance calculation
        numeric_values = []
        for score_dict in all_scores:
            label = score_dict.get("label", "neutral").lower()
            confidence = score_dict.get("score", 0.5)
            
            if label == "positive":
                numeric_values.append(confidence)
            elif label == "negative":
                numeric_values.append(-confidence)
            else:
                numeric_values.append(0.0)
        
        if len(numeric_values) < 2:
            return 0.5
        
        # Confidence is inverse of variance (higher consistency = higher confidence)
        std_dev = np.std(numeric_values)
        # Normalize: lower std_dev = higher confidence
        confidence = max(0.0, min(1.0, 1.0 - std_dev))
        
        return confidence
