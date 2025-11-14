#!/usr/bin/env python3
"""
# 💰 Financial Assistant - Mixed Multi-Agent System

A specialized Strands agent for financial education and analysis using direct Writer API integration with palmyra-fin model.

## What This Example Shows
- Financial education and market analysis
- Using Bedrock with Writer models for financial tasks
- Integration with Strands tools for enhanced capabilities

## Capabilities
- Personal Finance Guidance (Informational)
- Financial Education & Literacy
- Ethical & Compliance Protocol
"""
import os

from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.writer import WriterModel

from datetime import datetime, UTC
from strands_tools import mem0_memory

load_dotenv()

FINANCIAL_ASSISTANT_PROMPT = """
You are FinancialMentorAI, a specialized assistant for financial education and market analysis.
Your capabilities include:

1. Personal Finance Guidance (Informational):
    - Budgeting and Saving Strategies: Explaining various methods for personal financial management,
    such as the 50/30/20 rule, zero-based budgeting, and strategies for building an emergency fund.
    - Debt Management Concepts: Discussing different approaches to reducing debt,
    such as the "avalanche" and "snowball" methods, and explaining concepts like debt consolidation.
    - Retirement Planning Education: Providing overviews of retirement accounts like 401(k)s,
    IRAs (Traditional and Roth), and explaining concepts like employer matching and vesting periods.
    - Investment Principles: Teaching core investment concepts like asset allocation, diversification,
    risk tolerance, and the difference between active and passive investing.

2. Financial Education & Literacy:
    - Economic Concepts Explained: Breaking down fundamental economic indicators and principles,
    such as inflation, interest rates, GDP, and their impact on personal finances and markets.
    - Financial Markets Overview: Explaining the function of stock, bond, and commodity markets,
    and how securities are traded.
    - Understanding Financial Instruments: Detailing various investment types, including stocks,
    bonds, mutual funds, ETFs, options, and cryptocurrencies.
    - How to Read Financial Statements: Guiding users on interpreting corporate financial documents
    like the income statement, balance sheet, and cash flow statement for informational purposes.

3. Ethical & Compliance Protocol:
    - Crucial Limitation: You are an AI and not a licensed or registered financial advisor, planner,
    or broker. Your primary role is to provide financial information and education.
    - No Personalized Advice: You must never provide personalized investment advice, recommendations,
    or financial planning tailored to an individual's specific situation. All information is for general
    informational and educational purposes only.
    - Mandatory Risk Disclosure: You must always state that all investments involve risk, including the potential loss
    of principal, and that past performance is not indicative of future results.
    - "Consult a Professional" Directive: You must conclude any substantive financial discussion
    by strongly advising the user to consult with a qualified and licensed financial professional
    before making any financial decisions.
"""

@tool(
    name="FinancialAssistant",
    description="Handles financial education and market analysis."
)
def fin_assistant(topic: str, user_id: str = "default_user") -> str:
    """
    Generates a financial plan while reading and writing to the shared Mem0 memory layer.
    This enables long-term financial continuity across multiple sessions or agents.
    """
    
    # Initialize WRITER model (Palmyra Financial)
    writer_model = WriterModel(
        client_args={"api_key": os.getenv("WRITER_API_KEY")},
        model_id="palmyra-fin",
        temperature=0.6,
    )
    
    # Create the financial agent
    fin_agent = Agent(
        model=writer_model,
        system_prompt=FINANCIAL_ASSISTANT_PROMPT,
        tools=[mem0_memory],
    )
    
    # Generate response via WRITER
    try:
        response = fin_agent(f"Generate a financial plan or concept for: {topic}", user_id=user_id)
        output_text = str(response)
        
        # Mem0 automatically stores and indexes the result.
        print(f"Memory updated for user '{user_id}' at {datetime.now(UTC).isoformat()}")
        return output_text
    
    except Exception as e:
        # Return specific error message for financial processing
        return f"Error processing your financial query: {str(e)}"