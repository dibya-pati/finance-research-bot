import logging
import functools


def log_tool_call(tool_name, args):
    logging.getLogger("ResearchAgent").info(
        "[Tool Call] %s | args: %s", tool_name, args
    )


def log_tool_result(tool_name, result):
    logging.getLogger("ResearchAgent").info(
        "[Tool Result] %s | result: %s", tool_name, result
    )


# Tool description mapping for user-friendly progress messages
TOOL_PROGRESS_DESCRIPTIONS = {
    "ddg_search": "web search results from DuckDuckGo",
    "extract_article_text": "full article content from a web link",
    "get_news": "latest finance news headlines",
    "get_ohlc": "historical price (OHLC) data",
    "get_options_activity": "options chain data",
    "synthesize_report": "a synthesized research report",
}


def log_tool_execution(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        log_tool_call(tool_name, args if args else kwargs)
        # Print intermediate step to user (console)
        desc = TOOL_PROGRESS_DESCRIPTIONS.get(tool_name, f"results from {tool_name}")
        print(f"Collected {desc}...")
        result = func(*args, **kwargs)
        log_tool_result(tool_name, result)
        return result

    return wrapper
