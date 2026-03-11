# src/market_sentiment/tools/social_media_tool.py

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
import re
import os
from loguru import logger

from ..models import SocialPost, SocialCorpus

class SocialMediaToolInput(BaseModel):
    """Input schema for Social Media Tool."""
    query: str = Field(..., description="Search query (ticker, topic, or keywords)")
    platforms: List[str] = Field(default=["reddit", "twitter"], description="Platforms to search: reddit, twitter, stocktwits")
    max_results: int = Field(default=50, ge=1, le=100)
    time_window_hours: int = Field(default=24, ge=1, le=168)

class SocialMediaTool(BaseTool):
    name: str = "Social Media Sentiment Aggregator"
    description: str = (
        "Searches social media platforms (Reddit, Twitter, StockTwits) for posts and discussions "
        "about financial tickers or topics. Returns structured social media posts with engagement metrics. "
        "Filters spam and deduplicates posts."
    )
    args_schema: type[BaseModel] = SocialMediaToolInput

    def __init__(self):
        super().__init__()
        # Lazy initialization - set on first access
        self._serper = None
        self._serper_available = None

    @property
    def serper(self):
        """Lazy load SerperDevTool."""
        if self._serper is None:
            try:
                from crewai_tools import SerperDevTool
                # Verify API key is available
                if not os.getenv('SERPER_API_KEY'):
                    logger.warning("SERPER_API_KEY not found. SerperDevTool may fail with 403 error.")
                    self._serper_available = False
                else:
                    self._serper = SerperDevTool()
                    self._serper_available = True
            except ImportError:
                logger.warning("SerperDevTool not available")
                self._serper_available = False
            except Exception as e:
                logger.error(f"Failed to initialize SerperDevTool: {e}")
                self._serper_available = False
        return self._serper if self._serper_available else None

    @property
    def serper_available(self) -> bool:
        """Check if SerperDevTool is available."""
        if self._serper_available is None:
            _ = self.serper  # Trigger initialization
        return self._serper_available or False

    def _run(
        self,
        query: str,
        platforms: List[str] = None,
        max_results: int = 50,
        time_window_hours: int = 24
    ) -> SocialCorpus:
        """
        Aggregates social media posts from multiple platforms.
        
        Strategy:
        1. Search each platform using SerperDevTool with platform-specific queries
        2. Parse and structure posts
        3. Filter spam and deduplicate
        """
        if platforms is None:
            platforms = ["reddit", "twitter"]
        
        if not self.serper_available:
            logger.error("SerperDevTool not available")
            return SocialCorpus(
                posts=[],
                fetch_timestamp=datetime.now(),
                platforms=platforms
            )
        
        all_posts = []
        
        # Search each platform
        for platform in platforms:
            try:
                platform_posts = self._fetch_platform_posts(
                    platform, query, max_results // len(platforms), time_window_hours
                )
                all_posts.extend(platform_posts)
                logger.info(f"{platform} returned {len(platform_posts)} posts")
            except Exception as e:
                logger.warning(f"{platform} fetch failed: {e}")
        
        # Filter spam and deduplicate
        validated_posts = self._filter_and_deduplicate(all_posts)
        
        # Limit to max_results
        validated_posts = validated_posts[:max_results]
        
        return SocialCorpus(
            posts=validated_posts,
            fetch_timestamp=datetime.now(),
            platforms=platforms
        )
    
    def _fetch_platform_posts(
        self, platform: str, query: str, max_results: int, time_window: int
    ) -> List[SocialPost]:
        """Fetch posts from a specific platform."""
        platform_queries = {
            "reddit": f"site:reddit.com {query}",
            "twitter": f"site:twitter.com OR site:x.com {query}",
            "stocktwits": f"site:stocktwits.com {query}"
        }
        
        search_query = platform_queries.get(platform.lower(), query)
        
        try:
            if not self.serper_available or self.serper is None:
                return []
            # Fix: Use _run() with query as keyword argument
            result = self.serper._run(query=search_query)
            posts = []
            
            # Parse SerperDevTool response
            if isinstance(result, dict):
                organic_results = result.get('organic', []) or result.get('results', [])
                
                for item in organic_results[:max_results]:
                    try:
                        # Extract post ID from URL
                        post_id = self._extract_post_id(item.get('link', ''), platform)
                        
                        # Calculate engagement score (normalized)
                        engagement_score = self._calculate_engagement_score(item)
                        
                        # Extract ticker mentions
                        text = f"{item.get('title', '')} {item.get('snippet', '')}"
                        ticker_mentions = self._extract_tickers(text)
                        
                        # Parse timestamp
                        timestamp = self._parse_timestamp(item.get('date', ''))
                        
                        post = SocialPost(
                            platform=platform.lower(),
                            post_id=post_id or item.get('link', '')[:100],
                            snippet=item.get('snippet', '')[:300],
                            timestamp=timestamp,
                            ticker_mentions=ticker_mentions,
                            engagement_score=engagement_score
                        )
                        posts.append(post)
                    except Exception as e:
                        logger.warning(f"Failed to parse {platform} post: {e}")
                        continue
            elif isinstance(result, str):
                logger.warning(f"SerperDevTool returned string for {platform}: {result[:200]}")
            
            return posts
        except Exception as e:
            logger.error(f"Error fetching {platform} posts: {e}")
            return []
    
    def _extract_post_id(self, url: str, platform: str) -> str:
        """Extract post ID from platform URL."""
        try:
            if platform.lower() == "reddit":
                # Reddit URL: https://reddit.com/r/subreddit/comments/abc123/title/
                match = re.search(r'/comments/([a-z0-9]+)', url)
                return match.group(1) if match else url.split('/')[-1]
            elif platform.lower() == "twitter":
                # Twitter URL: https://twitter.com/user/status/1234567890
                match = re.search(r'/status/(\d+)', url)
                return match.group(1) if match else url.split('/')[-1]
            elif platform.lower() == "stocktwits":
                # StockTwits URL: https://stocktwits.com/symbol/AAPL or /message/123456
                match = re.search(r'/message/(\d+)', url)
                return match.group(1) if match else url.split('/')[-1]
        except:
            pass
        return url[:50]  # Fallback to URL fragment
    
    def _calculate_engagement_score(self, item: dict) -> float:
        """Calculate normalized engagement score from Serper result."""
        # Serper doesn't provide engagement metrics directly
        # Use snippet length and position as proxy
        snippet = item.get('snippet', '')
        position = item.get('position', 100)
        
        # Normalize: longer snippets and higher positions = higher engagement
        length_score = min(len(snippet) / 500, 1.0)  # Max 1.0 for 500+ chars
        position_score = max(0, (100 - position) / 100)  # Higher position = lower score
        
        # Combined score (weighted)
        engagement = (length_score * 0.6 + position_score * 0.4)
        return round(engagement, 3)
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract ticker symbols from text."""
        pattern = r'\$?([A-Z]{1,5})\b'
        matches = re.findall(pattern, text)
        
        false_positives = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        tickers = [m for m in matches if m not in false_positives]
        return list(set(tickers))
    
    def _parse_timestamp(self, date_str: str) -> datetime:
        """Parse timestamp from various formats."""
        if not date_str:
            return datetime.now()
        
        try:
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return datetime.now()
    
    def _filter_and_deduplicate(self, posts: List[SocialPost]) -> List[SocialPost]:
        """Filter spam and deduplicate posts."""
        if not posts:
            return []
        
        # Filter spam (low engagement)
        filtered = [p for p in posts if p.engagement_score >= 0.01]
        
        # Deduplicate by post_id
        seen_ids = set()
        unique_posts = []
        
        for post in filtered:
            if post.post_id not in seen_ids:
                seen_ids.add(post.post_id)
                unique_posts.append(post)
        
        return unique_posts