from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.text_file_knowledge_source import (
    TextFileKnowledgeSource
)
from crewai.memory import LongTermMemory, ShortTermMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai_tools import SerperDevTool
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables before tool instantiation
# Path from: src/market_sentiment/crew.py -> repo-root/.env
env_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path=env_path)

# Import custom tools
from .tools import (
    FinancialNewsTool,
    FinancialDataTool,
    HistoricalDataTool,
    SocialMediaTool,
    SentimentAnalysisTool,
    RiskCalculationTool,
)

# Import models for structured outputs
from .models import MarketSignals

# Instantiate tools (shared instances) - after env vars are loaded
financial_news_tool = FinancialNewsTool()
financial_data_tool = FinancialDataTool()
historical_data_tool = HistoricalDataTool()
social_media_tool = SocialMediaTool()
sentiment_analysis_tool = SentimentAnalysisTool()
risk_calculation_tool = RiskCalculationTool()

# SerperDevTool will automatically use SERPER_API_KEY from environment
# Verify it's available, log warning if not
if not os.getenv('SERPER_API_KEY'):
    import warnings
    warnings.warn("SERPER_API_KEY not found in environment. SerperDevTool may fail with 403 error.")
serper_dev_tool = SerperDevTool()


@CrewBase
class MarketSentiment():
    """MarketSentiment crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    @agent
    def data_scout(self) -> Agent:
        return Agent(
            config=self.agents_config['data_scout'],
            verbose=True,
            tools=[financial_news_tool]
        )

    @agent
    def market_data_fetcher(self) -> Agent:
        return Agent(
            config=self.agents_config['market_data_fetcher'],
            verbose=True,
            tools=[financial_data_tool, historical_data_tool]
        )
    
    @agent
    def alt_data_scout(self) -> Agent:
        return Agent(
            config=self.agents_config['alt_data_scout'],
            verbose=True,
            tools=[serper_dev_tool, social_media_tool]
        )

    @agent
    def sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['sentiment_analyst'],
            verbose=True,
            tools=[sentiment_analysis_tool],
            max_retry_limit=3
        )
    
    @agent
    def risk_engine(self) -> Agent:
        return Agent(
            config=self.agents_config['risk_engine'],
            verbose=True,
            tools=[risk_calculation_tool]
        )
    
    @agent
    def signal_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config['signal_synthesizer'],
            verbose=True,
            max_retry_limit=3
        )
    # @agent
    # def alertmanager(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['AlertManager'],   
    #         verbose=True    
    #     )

    @task
    def ingest_news_data(self) -> Task:
        return Task(
            config=self.tasks_config['ingest_news_data'],   
        )
    
    @task
    def fetch_market_metrics(self) -> Task:
        return Task(
            config=self.tasks_config['fetch_market_metrics'],   
        )

    @task
    def ingest_social_sentiment(self) -> Task:
        return Task(
            config=self.tasks_config['ingest_social_sentiment'],   
        )

    @task
    def synthesize_sentiment(self) -> Task:
        return Task(
            config=self.tasks_config['synthesize_sentiment'],   
        )
    
    @task
    def calculate_risk_score(self) -> Task:
        return Task(
            config=self.tasks_config['calculate_risk_score'],   
        )
    
    @task
    def compile_market_signals(self) -> Task:
        return Task(
            config=self.tasks_config['compile_market_signals'],
            output_pydantic=MarketSignals,
        )
        
    # @task
    # def conditional_alert(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['ConditionalAlert'],   
    #     )
    
    
    @crew
    def crew(self) -> Crew:
        """Creates the MarketSentiment crew"""
        # Option 1: Hybrid approach - Use hierarchical for parallel ingestion
        # with SentimentAnalyst as master for Phase 2
        # manager = Agent(
        #     config=self.agents_config['sentiment_analyst'],  # Master for Phase 2
        #     allow_delegation=True
    # )
        # Create knowledge sources
        # TextFileKnowledgeSource automatically prepends "knowledge/" to paths
        # So we use paths relative to the knowledge directory
        # glossary_source = TextFileKnowledgeSource(
        #     file_paths=[
        #         "glossary/market_terms.md",
        #         "glossary/risk_taxonomy.md",
        #         "glossary/sector_classifications.md"
        #     ]
        # )
    
        # policy_source = TextFileKnowledgeSource(
        #     file_paths=[
        #         "policies/risk_policy.md",
        #         "policies/compliance_rules.md",
        #         "policies/data_quality_standards.md"
        #     ]
        # )

        return Crew(
            agents=self.agents,   
            tasks=self.tasks,   
            process=Process.sequential,
            # manager_agent=manager,
            verbose=True,
            memory=True,
            # knowledge_sources=[glossary_source, policy_source],
            # Optional: Custom memory configuration like stock_picker
        #     long_term_memory=LongTermMemory(
        #         storage=LTMSQLiteStorage(
        #             db_path="./memory/long_term_memory_storage.db"
        #         )
        #     ),
        #     short_term_memory=ShortTermMemory(
        #         storage=RAGStorage(
        #             embedder={
        #                 "provider": "openai",
        #                 "config": {"model": 'text-embedding-3-small'}
        #             },
        #             type="short_term",
        #             path="./memory/"
        #         )
        #     ),


        )
