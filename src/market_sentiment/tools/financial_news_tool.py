# src/market_sentiment/tools/financial_news_tool.py

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
import re
import os
from loguru import logger

# Fix import - use relative import or absolute
from ..models import NewsArticle, NewsCorpus

class FinancialNewsToolInput(BaseModel):
    """Input schema for Financial News Tool."""
    query: str = Field(..., description="Search query (ticker, topic, or keywords)")
    max_results: int = Field(default=10, ge=1, le=50)
    time_window_hours: int = Field(default=24, ge=1, le=168)

class FinancialNewsTool(BaseTool):
    name: str = "Financial News Aggregator"
    description: str = (
        "Searches and aggregates validated financial news from multiple sources "
        "(GNews, Yahoo Finance, Serper) for comprehensive market intelligence. "
        "Returns structured news articles with deduplication and validation."
    )
    args_schema: type[BaseModel] = FinancialNewsToolInput

    def __init__(self):
        super().__init__()
        # Lazy initialization - set on first access
        self._gnews = None
        self._yahoo_tool = None
        self._gnews_available = None
        self._yahoo_available = None
        self._provider_cooldowns: dict[str, datetime] = {}
        self._cache: dict[tuple[str, str, int, int], tuple[datetime, List[NewsArticle]]] = {}

    def _cache_key(
        self, provider: str, query: str, max_results: int, time_window_hours: int
    ) -> tuple[str, str, int, int]:
        return (provider, query.strip().upper(), max_results, time_window_hours)

    def _get_cached(
        self, provider: str, query: str, max_results: int, time_window_hours: int
    ) -> Optional[List[NewsArticle]]:
        key = self._cache_key(provider, query, max_results, time_window_hours)
        cached = self._cache.get(key)
        if not cached:
            return None
        cached_at, articles = cached
        if datetime.now() - cached_at > timedelta(minutes=15):
            self._cache.pop(key, None)
            return None
        logger.info(f"{provider} cache hit for {query}")
        return articles

    def _set_cached(
        self,
        provider: str,
        query: str,
        max_results: int,
        time_window_hours: int,
        articles: List[NewsArticle],
    ) -> List[NewsArticle]:
        key = self._cache_key(provider, query, max_results, time_window_hours)
        self._cache[key] = (datetime.now(), articles)
        return articles

    def _provider_ready(self, provider: str) -> bool:
        cooldown_until = self._provider_cooldowns.get(provider)
        if cooldown_until and cooldown_until > datetime.now():
            remaining = int((cooldown_until - datetime.now()).total_seconds())
            logger.warning(
                f"{provider} is cooling down after rate limiting; "
                f"skipping for {remaining}s"
            )
            return False
        return True

    def _set_provider_cooldown(self, provider: str, minutes: int = 15) -> None:
        self._provider_cooldowns[provider] = datetime.now() + timedelta(minutes=minutes)

    @property
    def gnews(self):
        """Lazy load GNews client."""
        if self._gnews is None:
            try:
                from gnews import GNews
                # GNews may support api_key parameter - check if available in env
                gnews_api_key = os.getenv('GNEWS_API_KEY')
                if gnews_api_key:
                    # Try with API key if available
                    try:
                        self._gnews = GNews(language='en', country='US', api_key=gnews_api_key)
                        logger.info("GNews initialized with API key")
                    except TypeError:
                        # If api_key parameter not supported, try without it
                        # Some versions of GNews read from environment automatically
                        self._gnews = GNews(language='en', country='US')
                        logger.info("GNews initialized (API key may be read from environment)")
                else:
                    # Initialize without API key (free tier)
                    self._gnews = GNews(language='en', country='US')
                    logger.info("GNews initialized without API key (free tier)")
                self._gnews_available = True
            except ImportError:
                logger.warning("GNews package not installed. Install with: pip install gnews")
                self._gnews_available = False
            except Exception as e:
                logger.error(f"Failed to initialize GNews: {e}")
                self._gnews_available = False
        return self._gnews if self._gnews_available else None

    @property
    def gnews_available(self) -> bool:
        """Check if GNews is available."""
        if self._gnews_available is None:
            _ = self.gnews  # Trigger initialization
        return self._gnews_available or False

    @property
    def yahoo_tool(self):
        """Lazy load Yahoo Finance tool."""
        if self._yahoo_tool is None:
            try:
                from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
                self._yahoo_tool = YahooFinanceNewsTool()
                self._yahoo_available = True
            except ImportError:
                logger.warning("langchain_community not installed. Install with: pip install langchain-community")
                self._yahoo_available = False
        return self._yahoo_tool if self._yahoo_available else None

    @property
    def yahoo_available(self) -> bool:
        """Check if Yahoo Finance is available."""
        if self._yahoo_available is None:
            _ = self.yahoo_tool  # Trigger initialization
        return self._yahoo_available or False

    def _run(
        self, 
        query: str, 
        max_results: int = 10,
        time_window_hours: int = 24
    ) -> NewsCorpus:
        """
        Aggregates financial news from multiple validated sources.
        
        Strategy:
        1. Start with lower-cost / lower-rate-limit providers
        2. Use sentiment providers only if we still need more coverage
        3. Cache successful results to reduce repeated calls
        4. Deduplicate and validate articles
        """
        articles = []

        # 1. GNews gives decent breadth without burning premium credits first.
        if self.gnews_available:
            try:
                gnews_articles = self._fetch_gnews(query, max_results // 2, time_window_hours)
                articles.extend(gnews_articles)
                logger.info(f"GNews returned {len(gnews_articles)} articles")
            except Exception as e:
                logger.warning(f"GNews failed: {e}")

        # 2. AlphaVantage adds sentiment metadata with a generous free tier.
        if len(articles) < max_results * 2 / 3:
            try:
                av_articles = self._fetch_alphavantage(query, max_results // 3, time_window_hours)
                articles.extend(av_articles)
                logger.info(f"AlphaVantage returned {len(av_articles)} articles")
            except Exception as e:
                logger.warning(f"AlphaVantage failed: {e}")

        # 3. Finnhub adds company news without requiring the SDK.
        if len(articles) < max_results * 3 / 4:
            try:
                fh_articles = self._fetch_finnhub(query, max_results // 3, time_window_hours)
                articles.extend(fh_articles)
                logger.info(f"Finnhub returned {len(fh_articles)} articles")
            except Exception as e:
                logger.warning(f"Finnhub failed: {e}")

        # 4. Massive is valuable, but keep it late and light on the free tier.
        if len(articles) < max_results // 2:
            try:
                massive_articles = self._fetch_massive(query, min(max_results // 3, 3), time_window_hours)
                articles.extend(massive_articles)
                logger.info(f"Massive API returned {len(massive_articles)} articles with sentiment")
            except Exception as e:
                logger.warning(f"Massive API failed: {e}")

        # 5. Yahoo Finance (Ticker-specific news) - fallback if we still need more.
        if len(articles) < max_results // 2 and self.yahoo_available:
            try:
                yahoo_articles = self._fetch_yahoo_finance(query, max_results//4, time_window_hours)
                articles.extend(yahoo_articles)
                logger.info(f"Yahoo Finance returned {len(yahoo_articles)} articles")
            except Exception as e:
                logger.warning(f"Yahoo Finance failed: {e}")
        
        # 6. SerperDevTool (Broad coverage fallback) - Only if we still need more articles
        if len(articles) < max_results:
            try:
                remaining = max_results - len(articles)
                serper_articles = self._fetch_serper(query, remaining)
                articles.extend(serper_articles)
                logger.info(f"SerperDevTool returned {len(serper_articles)} articles")
            except Exception as e:
                logger.warning(f"SerperDevTool failed: {e}")
        
        # 4. Deduplicate and validate
        validated_articles = self._deduplicate_and_validate(articles)
        
        # 5. Limit to max_results
        validated_articles = validated_articles[:max_results]
        
        return NewsCorpus(
            articles=validated_articles,
            fetch_timestamp=datetime.now(),
            source="FinancialNewsTool (Multi-Source)"
        )

    def _fetch_massive(self, query: str, max_results: int, time_window: int) -> List[NewsArticle]:
        """Fetch from Massive API /v2/reference/news endpoint."""
        cached = self._get_cached("Massive", query, max_results, time_window)
        if cached is not None:
            return cached
        if not self._provider_ready("Massive"):
            return []

        try:
            from massive import RESTClient
            from massive.rest.models import TickerNews
            
            api_key = os.getenv('MASSIVE_API_KEY')
            if not api_key:
                logger.warning("MASSIVE_API_KEY not found")
                return []
            
            # Extract ticker from query
            ticker = self._extract_ticker_from_query(query)
            if not ticker:
                return []
            
            # Initialize Massive REST client
            client = RESTClient(api_key)
            
            # Calculate time range
            time_to = datetime.now()
            time_from = time_to - timedelta(hours=time_window)
            
            articles = []
            count = 0
            items_seen = 0
            max_items_seen = max(max_results * 3, 10)
            
            # Fetch news using list_ticker_news iterator
            for news_item in client.list_ticker_news(
                order="desc",  # Most recent first
                limit=str(min(max_results, 5)),
                sort="published_utc",
            ):
                try:
                    items_seen += 1
                    if items_seen > max_items_seen:
                        logger.info(
                            f"Massive scan budget reached for {ticker}; "
                            "stopping early to avoid free-tier pagination churn"
                        )
                        break

                    # Verify this is a TickerNews object
                    if not isinstance(news_item, TickerNews):
                        continue
                    
                    # Filter by ticker if the item has tickers attribute
                    # Skip if this article doesn't mention our ticker
                    item_tickers = getattr(news_item, 'tickers', None)
                    if item_tickers:
                        if isinstance(item_tickers, list):
                            if ticker not in item_tickers:
                                continue
                        elif isinstance(item_tickers, str):
                            if ticker != item_tickers:
                                continue
                    
                    # Parse timestamp and filter by time window
                    published_utc = getattr(news_item, 'published_utc', None)
                    if published_utc:
                        try:
                            # Handle string or datetime object
                            if isinstance(published_utc, str):
                                timestamp = datetime.fromisoformat(
                                    published_utc.replace('Z', '+00:00')
                                )
                            else:
                                timestamp = published_utc
                            
                            # Skip if outside time window
                            if timestamp < time_from or timestamp > time_to:
                                continue
                        except (ValueError, AttributeError, TypeError):
                            timestamp = datetime.now()
                    else:
                        timestamp = datetime.now()
                    
                    # Extract sentiment from insights (per-ticker)
                    sentiment_score = None
                    sentiment_reasoning = None
                    
                    # Access insights if available
                    if hasattr(news_item, 'insights') and news_item.insights:
                        for insight in news_item.insights:
                            # Check if insight has ticker attribute
                            insight_ticker = (
                                getattr(insight, 'ticker', None) or
                                getattr(insight, 'ticker_symbol', None)
                            )
                            if insight_ticker == ticker:
                                sentiment = (
                                    getattr(insight, 'sentiment', 'neutral') or
                                    'neutral'
                                )
                                # Convert to numeric: positive=1.0, negative=-1.0, neutral=0.0
                                if sentiment == 'positive':
                                    sentiment_score = 1.0
                                elif sentiment == 'negative':
                                    sentiment_score = -1.0
                                else:
                                    sentiment_score = 0.0
                                sentiment_reasoning = (
                                    getattr(insight, 'sentiment_reasoning', '') or
                                    getattr(insight, 'reasoning', '')
                                )
                                break
                    
                    # Get publisher name
                    publisher = getattr(news_item, 'publisher', None)
                    if publisher:
                        source_name = (
                            getattr(publisher, 'name', None) or
                            str(publisher) if isinstance(publisher, str) else
                            'Unknown'
                        )
                    else:
                        source_name = 'Unknown'
                    
                    # Get article URL
                    article_url = (
                        getattr(news_item, 'article_url', None) or
                        getattr(news_item, 'url', None) or
                        ''
                    )
                    
                    # Get description/snippet
                    description = (
                        getattr(news_item, 'description', None) or
                        getattr(news_item, 'summary', None) or
                        ''
                    )
                    
                    # Get tickers
                    tickers = (
                        getattr(news_item, 'tickers', None) or
                        [ticker]
                    )
                    if not isinstance(tickers, list):
                        tickers = [ticker]
                    
                    article = NewsArticle(
                        source=source_name,
                        title=getattr(news_item, 'title', '')[:200],
                        url=article_url,
                        snippet=description[:500],
                        timestamp=timestamp,
                        ticker_mentions=tickers,
                        toxicity_score=0.0,  # Massive doesn't provide this
                        sentiment_score=sentiment_score,
                        sentiment_reasoning=sentiment_reasoning
                    )
                    articles.append(article)
                    count += 1
                    
                    # Log sentiment if available
                    if sentiment_reasoning:
                        logger.debug(
                            f"Massive sentiment for {ticker}: "
                            f"{sentiment} - {sentiment_reasoning[:100]}"
                        )
                    
                    # Stop if we've reached max_results
                    if count >= max_results:
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to parse Massive article: {e}")
                    continue
            
            return self._set_cached("Massive", query, max_results, time_window, articles)
        except ImportError:
            logger.warning(
                "massive package not installed. "
                "Install with: pip install massive"
            )
            return []
        except Exception as e:
            if "429" in str(e):
                self._set_provider_cooldown("Massive", minutes=15)
                logger.warning(
                    "Massive rate limited the request. Cooling down and "
                    "relying on alternate providers instead."
                )
            logger.error(f"Massive API fetch error: {e}")
            return []
    
    def _fetch_alphavantage(
        self, query: str, max_results: int, time_window: int
    ) -> List[NewsArticle]:
        """Fetch from AlphaVantage NEWS_SENTIMENT API."""
        cached = self._get_cached("AlphaVantage", query, max_results, time_window)
        if cached is not None:
            return cached

        try:
            import requests
            
            api_key = os.getenv('ALPHAVANTAGE_API_KEY')
            if not api_key:
                logger.warning("ALPHAVANTAGE_API_KEY not found")
                return []
            
            # Extract ticker from query
            ticker = self._extract_ticker_from_query(query)
            if not ticker:
                return []
            
            # Calculate time range
            time_to = datetime.now()
            time_from = time_to - timedelta(hours=time_window)
            
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'time_from': time_from.strftime('%Y%m%dT%H%M'),
                'limit': min(max_results, 1000),
                'apikey': api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            feed = data.get('feed', [])
            
            for item in feed[:max_results]:
                try:
                    # AlphaVantage provides overall sentiment score
                    overall_sentiment = item.get('overall_sentiment_score', 0.0)
                    # Normalize to -1 to 1 range (AlphaVantage uses different scale)
                    sentiment_score = max(-1.0, min(1.0, overall_sentiment / 100.0))
                    
                    # Parse timestamp
                    time_published = item.get('time_published', '')
                    try:
                        # Format: YYYYMMDDTHHMM
                        timestamp = datetime.strptime(time_published, '%Y%m%dT%H%M')
                    except (ValueError, AttributeError):
                        timestamp = datetime.now()
                    
                    # Extract tickers from ticker_sentiment array
                    ticker_mentions = [ticker]  # Default to query ticker
                    if item.get('ticker_sentiment'):
                        ticker_mentions = [
                            ts.get('ticker')
                            for ts in item.get('ticker_sentiment', [])
                            if ts.get('ticker')
                        ] or [ticker]
                    
                    article = NewsArticle(
                        source=item.get('source', 'Unknown'),
                        title=item.get('title', '')[:200],
                        url=item.get('url', ''),
                        snippet=item.get('summary', '')[:500],
                        timestamp=timestamp,
                        ticker_mentions=ticker_mentions,
                        toxicity_score=0.0,
                        sentiment_score=sentiment_score,
                        sentiment_reasoning=None
                    )
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse AlphaVantage article: {e}")
                    continue
            
            return self._set_cached("AlphaVantage", query, max_results, time_window, articles)
        except Exception as e:
            logger.error(f"AlphaVantage fetch error: {e}")
            return []
    
    def _fetch_finnhub(
        self, query: str, max_results: int, time_window: int
    ) -> List[NewsArticle]:
        """Fetch from Finnhub Company News API."""
        cached = self._get_cached("Finnhub", query, max_results, time_window)
        if cached is not None:
            return cached

        try:
            import requests
            
            api_key = os.getenv('FINNHUB_API_KEY')
            if not api_key:
                logger.warning("FINNHUB_API_KEY not found")
                return []
            
            # Extract ticker from query
            ticker = self._extract_ticker_from_query(query)
            if not ticker:
                return []

            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(hours=time_window)

            response = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": ticker,
                    "from": from_date.strftime('%Y-%m-%d'),
                    "to": to_date.strftime('%Y-%m-%d'),
                    "token": api_key,
                },
                timeout=10,
            )
            response.raise_for_status()
            news = response.json()
            
            articles = []
            for item in news[:max_results]:
                try:
                    # Parse timestamp (Finnhub uses UNIX timestamp)
                    timestamp = datetime.fromtimestamp(item.get('datetime', 0))
                    
                    article = NewsArticle(
                        source=item.get('source', 'Unknown'),
                        title=item.get('headline', '')[:200],
                        url=item.get('url', ''),
                        snippet=item.get('summary', '')[:500],
                        timestamp=timestamp,
                        ticker_mentions=[ticker],
                        toxicity_score=0.0,
                        sentiment_score=None,  # Finnhub doesn't provide sentiment
                        sentiment_reasoning=None
                    )
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse Finnhub article: {e}")
                    continue
            
            return self._set_cached("Finnhub", query, max_results, time_window, articles)
        except Exception as e:
            logger.error(f"Finnhub fetch error: {e}")
            return []

    def _fetch_gnews(self, query: str, max_results: int, time_window: int) -> List[NewsArticle]:
        """Fetch from GNews API (financial-focused)."""
        if not self.gnews_available or self.gnews is None:
            return []
        
        try:
            # Add financial context to query
            financial_query = f"{query} finance stock market"
            self.gnews.max_results = max_results
            
            # Get news articles
            raw_articles = self.gnews.get_news(financial_query)
            
            articles = []
            for item in raw_articles[:max_results]:
                try:
                    # Parse timestamp
                    published_date = item.get('published date', '')
                    timestamp = self._parse_gnews_timestamp(published_date)
                    
                    # Extract ticker mentions from title and description
                    text = f"{item.get('title', '')} {item.get('description', '')}"
                    ticker_mentions = self._extract_tickers(text)
                    
                    article = NewsArticle(
                        source=item.get('source', {}).get('name', 'Unknown'),
                        title=item.get('title', '')[:200],  # Limit title length
                        url=item.get('url', ''),
                        snippet=item.get('description', '')[:500],  # Limit snippet
                        timestamp=timestamp,
                        ticker_mentions=ticker_mentions,
                        toxicity_score=0.0,
                        sentiment_score=None,  # GNews doesn't provide sentiment
                        sentiment_reasoning=None
                    )
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse GNews article: {e}")
                    continue
            
            return articles
        except Exception as e:
            logger.error(f"GNews fetch error: {e}")
            return []
    
    def _fetch_yahoo_finance(self, query: str, max_results: int, time_window: int) -> List[NewsArticle]:
        """Fetch from Yahoo Finance News using LangChain tool."""
        if not self.yahoo_available or self.yahoo_tool is None:
            return []
        
        try:
            # Extract potential ticker from query
            ticker = self._extract_ticker_from_query(query)
            if not ticker:
                # If no ticker found, try the query as-is
                ticker = query.upper().strip()
            
            # Use YahooFinanceNewsTool
            result = self.yahoo_tool.invoke(ticker)
            
            articles = []
            # YahooFinanceNewsTool returns a list of news items
            if isinstance(result, list):
                for item in result[:max_results]:
                    try:
                        # Parse Yahoo Finance news format
                        timestamp = self._parse_yahoo_timestamp(item.get('providerPublishTime', datetime.now()))
                        
                        text = f"{item.get('title', '')} {item.get('summary', '')}"
                        ticker_mentions = self._extract_tickers(text)
                        
                        article = NewsArticle(
                            source=item.get('publisher', 'Yahoo Finance'),
                            title=item.get('title', '')[:200],
                            url=item.get('link', ''),
                            snippet=item.get('summary', '')[:500],
                            timestamp=timestamp,
                            ticker_mentions=ticker_mentions,
                            toxicity_score=0.0,
                            sentiment_score=None,  # Yahoo doesn't provide sentiment
                            sentiment_reasoning=None
                        )
                        articles.append(article)
                    except Exception as e:
                        logger.warning(f"Failed to parse Yahoo Finance article: {e}")
                        continue
            elif isinstance(result, str):
                # If tool returns string, try to parse it
                logger.warning(f"Yahoo Finance returned string instead of list: {result[:100]}")
            
            return articles
        except Exception as e:
            logger.error(f"Yahoo Finance fetch error: {e}")
            return []
    
    def _fetch_serper(self, query: str, max_results: int) -> List[NewsArticle]:
        """Fetch from SerperDevTool as fallback."""
        try:
            from crewai_tools import SerperDevTool
            
            # Verify API key is available
            if not os.getenv('SERPER_API_KEY'):
                logger.warning("SERPER_API_KEY not found. SerperDevTool may fail.")
                return []
            
            serper = SerperDevTool()
            # Add financial/news context
            financial_query = f"{query} financial news"
            # Fix: Use _run() with query as keyword argument
            result = serper._run(query=financial_query)
            
            articles = []
            
            # SerperDevTool returns different formats, handle both dict and string
            if isinstance(result, dict):
                # Parse Serper response format
                news_results = result.get('news', []) or result.get('organic', [])
                
                for item in news_results[:max_results]:
                    try:
                        timestamp = self._parse_serper_timestamp(item.get('date', ''))
                        text = f"{item.get('title', '')} {item.get('snippet', '')}"
                        ticker_mentions = self._extract_tickers(text)
                        
                        article = NewsArticle(
                            source=item.get('source', 'Unknown'),
                            title=item.get('title', '')[:200],
                            url=item.get('link', ''),
                            snippet=item.get('snippet', '')[:500],
                            timestamp=timestamp,
                            ticker_mentions=ticker_mentions,
                            toxicity_score=0.0,
                            sentiment_score=None,  # Serper doesn't provide sentiment
                            sentiment_reasoning=None
                        )
                        articles.append(article)
                    except Exception as e:
                        logger.warning(f"Failed to parse Serper article: {e}")
                        continue
            elif isinstance(result, str):
                # If Serper returns string, log it
                logger.warning(f"SerperDevTool returned string: {result[:200]}")
            
            return articles
        except Exception as e:
            logger.error(f"SerperDevTool fetch error: {e}")
            return []
    
    def _deduplicate_and_validate(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicates and validate article quality."""
        if not articles:
            return []
        
        # Deduplicate by URL (primary key)
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            url = article.url.lower().strip()
            if url and url not in seen_urls:
                seen_urls.add(url)
                
                # Validate article quality
                if self._is_valid_article(article):
                    unique_articles.append(article)
        
        # Secondary deduplication by title similarity (simple)
        final_articles = []
        seen_titles = set()
        
        for article in unique_articles:
            title_key = article.title.lower().strip()[:50]  # First 50 chars
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                final_articles.append(article)
        
        return final_articles
    
    def _is_valid_article(self, article: NewsArticle) -> bool:
        """Validate article quality."""
        # Filter out low-quality articles
        if not article.title or len(article.title) < 10:
            return False
        
        if not article.url or not article.url.startswith('http'):
            return False
        
        # Filter high toxicity (if available)
        if article.toxicity_score > 0.7:
            return False
        
        return True
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract ticker symbols from text (1-5 uppercase letters)."""
        # Pattern: 1-5 uppercase letters, possibly with $ prefix
        pattern = r'\$?([A-Z]{1,5})\b'
        matches = re.findall(pattern, text)
        
        # Filter common false positives
        false_positives = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        tickers = [m for m in matches if m not in false_positives]
        return list(set(tickers))  # Remove duplicates
    
    def _extract_ticker_from_query(self, query: str) -> Optional[str]:
        """Extract ticker symbol from query string."""
        tickers = self._extract_tickers(query)
        return tickers[0] if tickers else None
    
    def _parse_gnews_timestamp(self, date_str: str) -> datetime:
        """Parse GNews timestamp."""
        if not date_str:
            return datetime.now()
        
        try:
            # GNews format: "2024-11-15 10:30:00"
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return datetime.now()
    
    def _parse_yahoo_timestamp(self, timestamp: any) -> datetime:
        """Parse Yahoo Finance timestamp."""
        if isinstance(timestamp, (int, float)):
            # Unix timestamp
            return datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            try:
                from dateutil import parser
                return parser.parse(timestamp)
            except:
                return datetime.now()
        return datetime.now()
    
    def _parse_serper_timestamp(self, date_str: str) -> datetime:
        """Parse Serper timestamp."""
        if not date_str:
            return datetime.now()
        
        try:
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return datetime.now()
