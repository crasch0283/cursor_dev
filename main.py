import os
import json
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
import datetime

# Load environment variable
load_dotenv()

# Initialize the OpenAI language model
llm = ChatOpenAI(model_name="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)

# Define agents
financial_analyst = Agent(
    role='Financial Analyst',
    goal='Provide accurate financial analysis and stock recommendations',
    backstory='You are an experienced financial analyst with a strong track record in stock market analysis.',
    verbose=True,
    llm=llm
)

market_researcher = Agent(
    role='Market Researcher',
    goal='Gather and analyze market trends and company information',
    backstory='You are a skilled market researcher with a keen eye for industry trends and company performance indicators.',
    verbose=True,
    llm=llm
)

investment_advisor = Agent(
    role='Investment Advisor',
    goal='Provide actionable investment advice based on analysis and research',
    backstory='You are a seasoned investment advisor known for your balanced and insightful recommendations.',
    verbose=True,
    llm=llm
)

# Define tasks
def research_task(stock_ticker):
    return Task(
        description=f'Research and gather key information about {stock_ticker}, including recent news, financial statements, and market position.',
        expected_output="Detailed analysis of the stock's performance indicators.",
        agent=market_researcher
    )

def analysis_task(stock_ticker):
    return Task(
        description=f'Analyze the gathered information for {stock_ticker}, assess the stock\'s financial health, and identify potential risks and opportunities.',
        expected_output="Expected analysis results including financial health, risks, and opportunities.",
        agent=financial_analyst
    )

def recommendation_task(stock_ticker):
    return Task(
        description=f'Based on the research and analysis for {stock_ticker}, provide a comprehensive investment recommendation.',
        agent=investment_advisor,
        expected_output="A detailed report with investment recommendations and analysis.",
        output_file=f"{stock_ticker}_analysis.md"
    )

# Main function to run the stock analyzer
def analyze_stock(ticker):
    print(f"Analyzing stock: {ticker}")

    # Create tasks with the specific stock ticker
    tasks = [
        research_task(ticker),
        analysis_task(ticker),
        recommendation_task(ticker)
    ]

    # Create the crew with the specific tasks
    stock_analysis_crew = Crew(
        agents=[market_researcher, financial_analyst, investment_advisor],
        tasks=tasks,
        verbose=True,
        iteration_limit=100,
        time_limit=300,
    )

    stock_analysis_crew.kickoff()
    

    print(f"Analysis complete. Results saved in {ticker}_analysis.md")

if __name__ == "__main__":
    stock_ticker = input("Enter the stock ticker symbol to analyze: ")
    analyze_stock(stock_ticker)