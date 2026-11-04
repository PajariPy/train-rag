"""Count the Gemini tokens a run spends, so every run reports what it cost."""

import threading

from google.genai import types

# Gemini 3.8 Flash standard prices, in dollars per million tokens. Thinking tokens are
# billed as output.
INPUT_PRICE = 0.75
OUTPUT_PRICE = 3.75


class UsageMeter:
    """Adds up the tokens of Gemini responses; safe to share between worker threads."""

    def __init__(self) -> None:
        """Start from zero."""
        self.lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.thinking_tokens = 0
        self.searches = 0

    def add(self, response: types.GenerateContentResponse) -> None:
        """Count one response's tokens.

        Args:
            response: A Gemini response.
        """
        usage = response.usage_metadata
        if usage is None:
            return

        with self.lock:
            self.calls += 1
            self.input_tokens += usage.prompt_token_count or 0
            self.output_tokens += usage.candidates_token_count or 0
            self.thinking_tokens += usage.thoughts_token_count or 0

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

        return (self.input_tokens * INPUT_PRICE + output * OUTPUT_PRICE) / 1_000_000

    def report(self) -> str:
        """Summarize the spend in one line.

        Returns:
            Calls, tokens and cost.
        """
        searches = f", {self.searches:,} searches" if self.searches else ""

        return (
            f"Gemini: {self.calls:,} calls{searches}, {self.input_tokens:,} input, "
            f"{self.output_tokens:,} output and {self.thinking_tokens:,} thinking tokens, "
            f"about ${self.cost():.2f}"
        )
