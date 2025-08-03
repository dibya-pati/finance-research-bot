import logging
import os
from datetime import datetime
from typing import Annotated, Any, Dict, List

import uuid
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,  # or define your own if unavailable
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field


from yfinance_tools import (
    extract_article_text,
    get_news,
    get_ohlc,
    get_options_by_ticker,
    ddg_search,
)

from logging_utils import log_tool_call, log_tool_result


# --- Logger Utility ---
def get_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    log_path = os.path.join(log_dir, log_filename)
    logger = logging.getLogger("ResearchAgent")
    logger.setLevel(logging.DEBUG)
    # Remove all handlers if already present (avoid duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


logger = get_logger()

# --- Load environment variables ---
load_dotenv()
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# --- State Model ---
class State(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    thread_id: str = ""
    user_id: str = ""


# --- Memory Store ---
memory_store = InMemoryStore()


def log_message_trail(messages, context=""):
    logger.info("[Message Trail%s]\n%s", f" {context}" if context else "", messages)


def log_llm_response(response, tokens=None):
    logger.info("[LLM Response] %s", response)
    if tokens is not None:
        logger.info("[Token Usage] %s", tokens)


@tool(
    description="""
    Generate a comprehensive, LLM-powered research report for a stock ticker.
    This tool combines fundamental data, recent news, and price action to answer a user research question.
    Inputs:
      - ticker (str): Stock ticker symbol (e.g., 'AAPL').
      - user_query (str): User's research question or topic of interest.
      - fundamentals (Dict[str, Any]): Fundamental financial data for the company.
      - news (List[str]): Recent news headlines or summaries relevant to the ticker.
      - ohlc (Dict[str, Any]): Recent OHLC (open, high, low, close, volume) price data.
    Output:
      - str: A synthesized research report text, including a clear BUY/SELL/HOLD recommendation, price range, and rationale.
    Use this tool to provide an end-to-end, up-to-date stock analysis that incorporates multiple data sources and LLM reasoning.
    """
)
def synthesize_report(
    ticker: str,
    user_query: str,
    fundamentals: Dict[str, Any],
    news: List[str],
    ohlc: Dict[str, Any],
) -> str:
    log_tool_call(
        "synthesize_report",
        {
            "ticker": ticker,
            "user_query": user_query,
            "fundamentals": fundamentals,
            "news": news,
            "ohlc": ohlc,
        },
    )

    def format_fundamentals(f):
        if not f:
            return "Key Fundamentals: No fundamental data available."
        return (
            f"Key Fundamentals:\n"
            f"- Name: {f.get('Name', 'N/A')}\n"
            f"- Market Cap: {f.get('MarketCapitalization', 'N/A')}\n"
            f"- PE Ratio: {f.get('PERatio', 'N/A')}\n"
            f"- 52 Week High: {f.get('52WeekHigh', 'N/A')}\n"
            f"- 52 Week Low: {f.get('52WeekLow', 'N/A')}\n"
            f"- Sector: {f.get('Sector', 'N/A')}\n"
            f"- Industry: {f.get('Industry', 'N/A')}\n"
            f"- EPS: {f.get('EPS', 'N/A')}\n"
            f"- Revenue TTM: {f.get('RevenueTTM', 'N/A')}"
        )

    def format_news(news_list):
        if not news_list:
            return "Recent News: No recent news articles found."
        return "Recent News Headlines:\n" + "\n".join(news_list)

    def analyze_ohlc(ohlc_data):
        if not ohlc_data:
            return {}
        try:
            sorteddates = sorted(ohlc_data.keys(), reverse=True)
            latestdate = sorteddates[0]
            oldestdate = sorteddates[-1]
            currentprice = float(ohlc_data[latestdate]["close"])
            oldestprice = float(ohlc_data[oldestdate]["close"])
            highs = [float(data["high"]) for data in ohlc_data.values()]
            lows = [float(data["low"]) for data in ohlc_data.values()]
            volumes = [float(data["volume"]) for data in ohlc_data.values()]
            monthreturn = (
                (currentprice - oldestprice) / oldestprice * 100 if oldestprice else 0
            )
            return {
                "currentprice": round(currentprice, 2),
                "monthhigh": round(max(highs), 2),
                "monthlow": round(min(lows), 2),
                "monthreturn": round(monthreturn, 2),
                "avgvolume": int(sum(volumes) / len(volumes)) if volumes else 0,
            }
        except Exception as e:
            logger.error("Error analyzing OHLC data: %s", e)
            return {}

    def format_price(ohlc_summary):
        if not ohlc_summary:
            return "Recent Price Action: No price data available."
        return (
            "Recent Price Action (Last 30 Days):\n"
            f"- Current Price: {ohlc_summary.get('currentprice', 'N/A')}\n"
            f"- 30-Day High: {ohlc_summary.get('monthhigh', 'N/A')}\n"
            f"- 30-Day Low: {ohlc_summary.get('monthlow', 'N/A')}\n"
            f"- 30-Day Return: {ohlc_summary.get('monthreturn', 'N/A')}%\n"
            f"- Average Volume: {ohlc_summary.get('avgvolume', 'N/A')}"
        )

    ohlc_summary = analyze_ohlc(ohlc)
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prompt = (
        f"You are a financial analyst agent. Analyze the following stock and answer the user's research question.\n"
        f"ANALYSIS DATE: {current_date}\n"
        f"Stock ticker: {ticker}\n"
        f"User Query: {user_query}\n\n"
        f"{format_fundamentals(fundamentals)}\n\n"
        f"{format_price(ohlc_summary)}\n\n"
        f"{format_news(news)}\n\n"
        "IMPORTANT: Start your response by confirming the analysis date and the last closing price to verify you are using the most recent data. "
        "Based on the above information, provide:\n"
        "1. A clear BUY, SELL, or HOLD recommendation.\n"
        "2. A suggested price range for action if applicable.\n"
        "3. A concise rationale explaining your reasoning.\n"
        "If data is limited, acknowledge the limitations and provide the best analysis possible with available information."
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", temperature=0, convert_system_message_to_human=True
    )  # Add this parameter
    response = llm.invoke([("user", prompt)])
    log_llm_response(response)
    log_tool_result("synthesize_report", response.content)
    return response.content


LANGGRAPH_TOOLS = [
    extract_article_text,
    get_news,
    get_ohlc,
    get_options_by_ticker,
    ddg_search,
]


def save_llm_html(html_content: str) -> str:
    """Save LLM-generated HTML directly to file."""
    import re
    import os

    # Ensure reports directory exists
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    # Try to extract ticker and timestamp from HTML comments first
    ticker = None
    timestamp = None
    ticker_match = re.search(r"<!-- TICKER: ([A-Za-z0-9_\-]+) -->", html_content)
    if ticker_match:
        ticker = ticker_match.group(1)
    else:
        filename_match = re.search(
            r"<!-- FILENAME: ([A-Za-z0-9_\-]+)_(\d{8}_\d{6})", html_content
        )
        if filename_match:
            ticker = filename_match.group(1)
            timestamp = filename_match.group(2)

    # If not found in comments, try to extract from <title>
    if not ticker or not timestamp:
        # Try to match <title>TICKER_YYYYMMDD_HHMMSS.html</title>
        title_match = re.search(
            r"<title>([A-Za-z0-9_\-]+)_(\d{8}_\d{6})\.html</title>", html_content
        )
        if not title_match:
            # Fallback: match <title>TICKER_YYYYMMDD_HHMMSS.html (without requiring </title>)
            title_match = re.search(
                r"<title>([A-Za-z0-9_\-]+)_(\d{8}_\d{6})\.html", html_content
            )
        if title_match:
            ticker = title_match.group(1)
            timestamp = title_match.group(2)

    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Compose filename: ticker_TIMESTAMP.html (fallback to report_TIMESTAMP.html)
    if ticker:
        filename = f"{ticker}_{timestamp}.html"
    else:
        filename = f"report_{timestamp}.html"

    file_path = os.path.join(reports_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return file_path


def ensure_string(obj):
    if isinstance(obj, list):
        return "\n".join(str(x) for x in obj)
    return str(obj)


def extract_and_save_dual_output(llm_response):
    import re

    llm_response = ensure_string(llm_response)

    # Try to extract both sections
    console_match = re.search(
        r"CONSOLE_SUMMARY:\s*\n(.*?)(?=HTML_REPORT:|$)", llm_response, re.DOTALL
    )
    html_match = re.search(r"HTML_REPORT:\s*\n(.*)", llm_response, re.DOTALL)

    # Default fallback: treat full response as console summary
    if not console_match and not html_match:
        return {"console": llm_response.strip(), "filename": None}

    console_summary = console_match.group(1).strip() if console_match else ""
    html_content = html_match.group(1).strip() if html_match else ""

    filename = None
    if html_content:
        filename = save_llm_html(html_content)

    # If only HTML is present, use it as console summary as well
    if not console_summary and html_content:
        console_summary = (
            html_content[:250] + "..." if len(html_content) > 250 else html_content
        )

    return {"console": console_summary or "Analysis complete", "filename": filename}


def get_prior_context(config: RunnableConfig):
    current_datetime = datetime.now()
    system_message_content = (
        f"The current date (YYYY-MM-DD) format is: {current_datetime.date()} and time is: {current_datetime.time()}\n"
        """
You are a Professional Financial Research Analyst. Analyze stocks using available tools and provide comprehensive investment insights.

NEWS SOURCING GUIDELINES:
- If the user requests or the context requires specific finance news, use the finance news tool (Yahoo Finance via get_news).
- For general, broad, or non-finance news, use the DuckDuckGo search tool.
- When using DuckDuckGo, you will receive URLs. If you find a relevant link and need to extract the full article content or HTML for deeper analysis (such as for news, research, or detailed information), you should call the get_article tool (extract_article_text) on that link.

ANALYSIS RULES:
1. General stock questions: Perform BOTH fundamental (news) and technical analysis
2. "Technical analysis" requests: Focus on technical indicators and patterns only
3. "Fundamental analysis" requests: Focus on news, earnings, and company fundamentals only
4. Always use ALL relevant tools available - never provide analysis without current data

WORKFLOW GUIDELINES:
- Short-term analysis (1-3 months): Call news + 3-month OHLC data
- Medium-term analysis (3-12 months): Call news + 1-year OHLC data  
- Long-term analysis (1+ years): Call news + 2-year OHLC data
- Options analysis: Include options activity data
- Call tools in parallel when possible for efficiency[1][6]

REQUIRED OUTPUT FORMAT:
- Investment Rating: BUY/SELL/HOLD with confidence level
- Bullish Case: Positive signals, catalysts, upside targets
- Bearish Case: Risks, concerns, downside levels
- Technical Summary: Chart patterns, indicators, support/resistance levels
- Risk Assessment: Key risks, stop-loss levels, position sizing

TECHNICAL ANALYSIS FOCUS:
- Chart patterns: Head & shoulders, flags, triangles, double tops/bottoms
- Indicators: Moving averages, RSI, MACD, Bollinger Bands, volume
- Support/resistance levels with specific price points

FUNDAMENTAL ANALYSIS FOCUS:
- Recent news sentiment and market impact
- Earnings trends and analyst expectations
- Sector performance and competitive position
- Market catalysts and risk factors

CRITICAL OUTPUT FORMAT:
You MUST structure your response in exactly two sections:

CONSOLE_SUMMARY:
[Provide a concise 2-3 sentence summary with investment rating for console display]

HTML_REPORT:
[Provide a COMPLETE, SELF-CONTAINED HTML document with:
- Proper DOCTYPE, head, and body tags
- Embedded CSS styling
- Ticker symbol in filename format as: TICKER_YYYYMMDD_HHMMSS.html
- All analysis content properly formatted]

The HTML_REPORT should be a complete, ready-to-save HTML file with no additional processing needed.
"""
    )

    thread_id = config["configurable"]["thread_id"]
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "research")

    prior_messages = memory_store.get(namespace, thread_id)
    if prior_messages and prior_messages.value:
        prior_messages.value = prior_messages.value[-5:]  # last 5 messages

        # Enhanced filtering: Remove empty messages and non-conversational types
        from langchain_core.messages import HumanMessage, AIMessage

        log_message_trail(
            prior_messages.value, context=f"user_id={user_id}, thread_id={thread_id}"
        )
        filtered_messages = []
        for msg in prior_messages.value:
            if isinstance(msg, (HumanMessage, AIMessage)) and msg.content:
                if isinstance(msg.content, str):
                    if msg.content.strip():
                        filtered_messages.append(msg)
                elif isinstance(msg.content, list):
                    # Keep if any element is a non-empty string
                    if any(isinstance(x, str) and x.strip() for x in msg.content):
                        filtered_messages.append(msg)

        prior_messages.value = filtered_messages

    # Create the system message properly
    system_message = SystemMessage(content=system_message_content)

    # Build the message list correctly - START WITH SYSTEM MESSAGE ONLY
    prior_messages_with_system = [system_message]

    if prior_messages and prior_messages.value:
        prior_messages_with_system.extend(prior_messages.value)

    # log_message_trail(
    #     prior_messages_with_system, context=f"user_id={user_id}, thread_id={thread_id}"
    # )

    return prior_messages_with_system


def create_research_agent_graph():
    """
    Creates and returns a compiled LangGraph research agent app with memory, thread/context management,
    and explicit tool descriptions for optimal supervisor reasoning and tool selection.
    """
    logger.info("Creating research agent graph.")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    llm_with_tools = llm.bind_tools(LANGGRAPH_TOOLS)

    def supervisor_node(state: State, config: RunnableConfig) -> dict:
        # Get the prior context with system message
        prior_messages = get_prior_context(config)

        # Combine with current state messages
        all_messages = prior_messages + state.messages

        # Invoke LLM with tools
        response = llm_with_tools.invoke(all_messages)
        log_llm_response(response)

        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("tools", ToolNode(tools=LANGGRAPH_TOOLS))
    graph.add_conditional_edges(
        "supervisor", tools_condition, {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "supervisor")
    graph.add_edge(START, "supervisor")
    logger.info("Research agent graph compiled.")
    return graph.compile()


# --- Main Chat Loop with Thread ID Support ---
def main():
    print("🤖 LangGraph Research Agent with Memory (thread-aware)")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("You can specify a thread ID to continue a previous conversation.")

    user_id = "djpati"
    thread_id = f"{str(datetime.now())}_{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    namespace = (user_id, "research")

    app = create_research_agent_graph()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            logger.info("Session ended by user_id=%s, thread_id=%s", user_id, thread_id)
            break

        try:
            logger.info("User input received: %s", user_input)

            # Get prior context (includes system message)
            prior_messages_with_system = get_prior_context(config)

            # Create user message properly
            from langchain_core.messages import HumanMessage

            user_message = HumanMessage(content=user_input)

            # Combine messages correctly
            all_messages = prior_messages_with_system + [user_message]

            result = app.invoke(
                {
                    "messages": all_messages,
                    "thread_id": thread_id,
                    "user_id": user_id,
                },
                config=config,
            )

            # Print the final answer
            final_message = result["messages"][-1]
            sections = extract_and_save_dual_output(final_message.content)
            print("Bot:", sections["console"])

            # Store the conversation in memory
            memory_store.put(namespace, thread_id, result["messages"])

        except Exception as e:
            logger.error("Error during conversation: %s", e)
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
