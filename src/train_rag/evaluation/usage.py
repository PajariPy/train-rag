"""Count the Gemini tokens a run spends, so every run reports what it cost."""

import threading

from google.genai import types

# Gemini 3.8 Flash standard prices, in dollars per million tokens. Thinking tokens are
# billed as output.
INPUT_PRICE = 0.75
OUTPUT_PRICE = 3.75
# Context caching. ASSUMED, not checked against this model's pricing page: cached input at
# 10% of the input price (the ratio of recent Flash models), writing a cache at the full
# input price, and storage at $1.00 per million tokens per hour.
CACHED_INPUT_PRICE = 0.075
CACHE_STORAGE_PRICE = 1.00


class UsageMeter:
    """Adds up the tokens of Gemini responses; safe to share between worker threads."""

    def __init__(self) -> None:
        """Start from zero."""
        self.lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.thinking_tokens = 0
        self.cached_tokens = 0
        self.cache_write_tokens = 0
        self.cache_token_hours = 0.0
        self.searches = 0

    def add(self, response: types.GenerateContentResponse) -> None:
        """Count one response's tokens.

        Args:
            response: A Gemini response.
        """
        usage = response.usage_metadata
        if usage is None:
            return

        cached = usage.cached_content_token_count or 0
        with self.lock:
            self.calls += 1
            self.input_tokens += (usage.prompt_token_count or 0) - cached
            self.cached_tokens += cached
            self.output_tokens += usage.candidates_token_count or 0
            self.thinking_tokens += usage.thoughts_token_count or 0

    def add_cache_write(self, tokens: int, ttl_seconds: int) -> None:
        """Count one context cache: writing its tokens, and storing them for its TTL.

        Args:
            tokens: Tokens the cache holds.
            ttl_seconds: How long it is kept; storage is billed for all of it.
        """
        with self.lock:
            self.cache_write_tokens += tokens
            self.cache_token_hours += tokens * ttl_seconds / 3600

    def add_searches(self, count: int) -> None:
        """Count knowledge-base searches an agent made.

        Args:
            count: Searches made.
        """
        with self.lock:
            self.searches += count

    def cost(self) -> float:
        """Price the counted tokens.

        Returns:
            Dollars at Gemini 3.8 Flash standard prices.
        """
        output = self.output_tokens + self.thinking_tokens
        tokens = (
            (self.input_tokens + self.cache_write_tokens) * INPUT_PRICE
            + self.cached_tokens * CACHED_INPUT_PRICE
            + output * OUTPUT_PRICE
            + self.cache_token_hours * CACHE_STORAGE_PRICE
        )

        return tokens / 1_000_000

    def report(self) -> str:
        """Summarize the spend in one line.

        Returns:
            Calls, tokens and cost.
        """
        searches = f", {self.searches:,} searches" if self.searches else ""
        cached = (
            f" (+{self.cached_tokens:,} cached, {self.cache_write_tokens:,} written to caches)"
            if self.cached_tokens or self.cache_write_tokens
            else ""
        )

        return (
            f"Gemini: {self.calls:,} calls{searches}, {self.input_tokens:,} input{cached}, "
            f"{self.output_tokens:,} output and {self.thinking_tokens:,} thinking tokens, "
            f"about ${self.cost():.2f}"
        )
