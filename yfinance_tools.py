from typing import Any, Dict, List, Optional, Union
import yfinance as yf
from langchain_core.tools import tool
from logging_utils import log_tool_execution
from newspaper import Article
from ddgs import DDGS
import os
import requests
import finnhub
import pandas as pd
import io


# --- DuckDuckGo Search Tool ---
@tool(
    description="""
    Perform a DuckDuckGo web search and return a list of results. Use this tool for any topic, question, or information need—especially when you do not have enough knowledge, context, or data from other tools, or when the user asks about something outside of finance or your training data. This is a general-purpose search tool for the latest information, news, or facts on any subject.

    If you want to follow up on a specific search result and extract the full article content or HTML from a link, use the extract_article_text tool with the URL from the search result.

    Parameters:
        query (str): The search query string.
        max_results (int): Maximum number of results to return (default: 5).
    Returns:
        Dict[str, Any]: Dictionary with a 'results' key containing a list of dicts with 'title', 'href', and 'body'.
    """
)
@log_tool_execution
def ddg_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title"),
                        "href": r.get("href"),
                        "body": r.get("body"),
                    }
                )
            return {"results": results}
    except Exception as e:
        return {"error": str(e)}


@tool
@log_tool_execution
def extract_article_text(url: str, fetch_html: bool = False) -> Dict[str, Any]:
    """
    Extracts article content and metadata from a given URL using newspaper3k.
    use this tool if you want more information about an article, and the description is not enough
    Parameters:
        url (str): The URL of the article to extract.
        fetch_html (bool): If True, include the raw HTML of the page in the result.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'title': Article title (str)
            - 'text': Main article text (str)
            - 'authors': List of authors (List[str])
            - 'publish_date': Publish date as ISO string, if available (str or None)
            - 'top_image': URL of the top image (str)
            - 'summary': Article summary (str)
            - 'source_url': The original URL (str)
            - 'html' (optional): Raw HTML of the page (str)
            - 'error': Error message (str), only if extraction fails

    Raises:
        None. All exceptions are caught and returned as an 'error' key.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        result = {
            "title": article.title,
            "text": article.text,
            "authors": article.authors,
            "publish_date": article.publish_date.isoformat()
            if article.publish_date
            else None,
            "top_image": article.top_image,
            "summary": article.summary,
            "source_url": url,
        }
        if fetch_html:
            result["html"] = article.html
        return result
    except Exception as e:
        return {"error": str(e), "source_url": url}


@tool
@log_tool_execution
def get_news(
    ticker_symbol: str, max_items: Optional[int] = 10
) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    """
    Retrieve recent news headlines for a ticker from Yahoo Finance.

    Parameters:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').
        max_items (Optional[int]): Maximum number of news items to return (default: 10).

    Returns:
        List[Dict[str, Any]]: List of news items with title, link, publisher, and publish time.
        Dict[str, str]: Error message if retrieval fails.
    """
    ticker = yf.Ticker(ticker_symbol)
    try:
        news_items = ticker.news
        return news_items[:max_items]
    except Exception as e:
        return {"error": str(e)}


@tool
@log_tool_execution
def get_ohlc(
    ticker_symbol: str,
    period: str = "1mo",
    interval: str = "1d",
    auto_adjust: bool = True,
    prepost: bool = False,
    actions: bool = True,
) -> Union[str, Dict[str, str]]:
    """
    Retrieve OHLC (Open, High, Low, Close, Volume) data for a ticker and return as a CSV string
    with all numeric values rounded to 2 decimal places.

    Parameters:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').
        period (str): Data period (e.g., '1d', '5d', '1mo', '3mo', '1y', 'max').
        interval (str): Data interval (e.g., '1m', '5m', '1h', '1d', '1wk', '1mo').
        auto_adjust (bool): Adjust all OHLC automatically (default: True).
        prepost (bool): Include Pre and Post market data? (default: False)
        actions (bool): Include dividends and splits? (default: True)

    Returns:
        str: CSV formatted string of OHLC data (rounded to 2 decimals).
        Dict[str, str]: Error message if retrieval fails.
    """
    ticker = yf.Ticker(ticker_symbol)
    try:
        df = ticker.history(
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            prepost=prepost,
            actions=actions,
        )
        df.reset_index(inplace=True)
        # Round all float columns to 2 decimal places
        float_cols = df.select_dtypes(include="float").columns
        df[float_cols] = df[float_cols].round(2)
        csv_data = df.to_csv(index=False)
        return csv_data
    except Exception as e:
        return {"error": str(e)}


# @tool
# @log_tool_execution
# def get_options_by_ticker(
#     ticker_symbol: str,
#     expiration: Optional[str] = None
# ) -> str:
#     """
#     Retrieve the options chain (calls and puts) for a given stock ticker and expiration date using Finnhub.

#     Parameters:
#         ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').
#         expiration (str, optional): Expiration date in 'YYYY-MM-DD' format. If None, uses the earliest available expiration.

#     Returns:
#         str: CSV-formatted string containing two sections: calls and puts, or an error message.

#     Environment:
#         FINNHUB_API_KEY: Your Finnhub API key.
#     """
#     # Step 1: Read API key
#     api_key = os.getenv("FINNHUB_API_KEY")
#     if not api_key:
#         return "Error: Finnhub API key not found in environment (FINNHUB_API_KEY)."
#     try:
#         finnhub_client = finnhub.Client(api_key=api_key)
#         chain = finnhub_client.option_chain(ticker_symbol)
#         if not chain or "data" not in chain or not chain["data"]:
#             return "No options data available for this ticker."
#         expirations = sorted(set(item["expirationDate"] for item in chain["data"]))
#         if not expirations:
#             return "No expirations found."
#         selected_exp = expiration if expiration in expirations else expirations[0]
#         options = [item for item in chain["data"] if item["expirationDate"] == selected_exp]
#         if not options:
#             return f"No options for expiration {selected_exp}."
#         calls = [o for o in options if o["type"] == "CALL"]
#         puts = [o for o in options if o["type"] == "PUT"]
#         calls_df = pd.DataFrame(calls)
#         puts_df = pd.DataFrame(puts)
#         calls_csv = io.StringIO()
#         puts_csv = io.StringIO()
#         calls_df.to_csv(calls_csv, index=False)
#         puts_df.to_csv(puts_csv, index=False)
#         result = (
#             f"# Expiration: {selected_exp}\n\n# Calls\n"
#             + calls_csv.getvalue()
#             + "\n# Puts\n"
#             + puts_csv.getvalue()
#         )
#         return result
#     except Exception as e:
#         return f"Error: {str(e)}"

# def get_top_options_activity(
#     limit: int = 20
# ) -> Union[List[Dict], str]:
#     """
#     Retrieve a list of the most active options contracts market-wide using Finnhub's unusual options endpoint.

#     Parameters:
#         limit (int, optional): The maximum number of top activity records to return (default: 20).

#     Returns:
#         list of dict: Each dict contains data for an active options contract (e.g., symbol, volume, type, etc.).
#         str: Error message if retrieval fails or data is unavailable.

#     Environment:
#         FINNHUB_API_KEY: Your Finnhub API key.
#     """
#     # Step 1: Read API key
#     api_key = os.getenv("FINNHUB_API_KEY")
#     if not api_key:
#         return "Error: Finnhub API key not found in environment (FINNHUB_API_KEY)."
#     try:
#         url = "https://finnhub.io/api/v1/stock/unusual-options"
#         params = {"token": api_key}
#         response = requests.get(url, params=params)
#         if response.status_code != 200:
#             return f"Error: {response.status_code} - {response.text}"
#         data = response.json()
#         if not data or "data" not in data or not data["data"]:
#             return "No unusual options activity data available."
#         return data["data"][:limit]
#     except Exception as e:
#         return f"Error: {str(e)}"


@tool
@log_tool_execution
def get_options_by_ticker(
    ticker_symbol: str, expiration: Optional[str] = None
) -> Union[str, Dict[str, str]]:
    """
    Retrieve options chain data for a ticker symbol and expiration date, and return as CSV string (calls and puts).

    Parameters:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').
        expiration (Optional[str]): Expiration date in 'YYYY-MM-DD' format. If None, uses the latest available.

    Returns:
        str: CSV formatted string with two sections: calls and puts (with headers).
        Dict[str, str]: Error message if retrieval fails.
    """
    import io

    ticker = yf.Ticker(ticker_symbol)
    try:
        expirations = ticker.options
        if not expirations:
            return {"error": "No options data available"}
        if expiration is None or expiration not in expirations:
            expiration = expirations[-1]  # Use the latest expiration date
        option_chain = ticker.option_chain(expiration)
        # Convert calls and puts DataFrames to CSV
        calls_csv = io.StringIO()
        puts_csv = io.StringIO()
        option_chain.calls.to_csv(calls_csv, index=False)
        option_chain.puts.to_csv(puts_csv, index=False)
        calls_csv_str = calls_csv.getvalue()
        puts_csv_str = puts_csv.getvalue()
        # Combine with section headers
        csv_combined = (
            f"# Expiration: {expiration}\n\n# Calls\n"
            + calls_csv_str
            + "\n# Puts\n"
            + puts_csv_str
        )
        return csv_combined
    except Exception as e:
        return {"error": str(e)}
