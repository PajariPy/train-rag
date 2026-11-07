"""A Gemini context cache of the start of one agent conversation.

Every agent turn resends the whole conversation. The cache stores its start once (the
instructions, the tools and the turns so far) so that later turns are billed for it at the
cached rate and send only what came after. A cache always holds a prefix: to cache more,
a new cache replaces the old one.

A cache fixes the tools and their mode, so a turn that must switch tools off can't use it;
the agent then sends the whole conversation uncached. Caching is an optimisation: if a
cache can't be made (too few tokens, an API error), the agent simply runs uncached.
"""

import logging

from google import genai
from google.genai import types

from train_rag.evaluation.pipeline_answer import GEMINI_MODEL
from train_rag.evaluation.usage import UsageMeter

TTL_SECONDS = 300  # longer than any question takes; storage is billed for the whole TTL
logger = logging.getLogger(__name__)


class ConversationCache:
    """Holds at most one cache of a conversation's start, and how many turns it covers."""

    def __init__(self, client: genai.Client, usage: UsageMeter) -> None:
        """Start with no cache.

        Args:
            client: Gemini client.
            usage: Counts the cache's writes and storage.
        """
        self.client = client
        self.usage = usage
        self.name: str | None = None
        self.covered = 0  # how many leading turns of the conversation the cache holds

    def store(
        self, instructions: str, tools: types.Tool, contents: list[types.Content]
    ) -> None:
        """Cache the conversation so far, replacing any older cache.

        Args:
            instructions: The system instructions.
            tools: The tools, fixed in automatic mode.
            contents: The turns to cache, from the start.
        """
        mode = types.FunctionCallingConfigMode.AUTO
        try:
            cache = self.client.caches.create(
                model=GEMINI_MODEL,
                config=types.CreateCachedContentConfig(
                    system_instruction=instructions,
                    tools=[tools],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode=mode)
                    ),
                    contents=list(contents),
                    ttl=f"{TTL_SECONDS}s",
                ),
            )
        except Exception as error:  # noqa: BLE001 - caching is optional; run uncached
            logger.info("Not caching the conversation: %s", error)
            return

        self.close()
        self.name = cache.name
        self.covered = len(contents)
        tokens = cache.usage_metadata.total_token_count if cache.usage_metadata else 0
        self.usage.add_cache_write(tokens or 0, TTL_SECONDS)

    def close(self) -> None:
        """Delete the cache, if there is one."""
        if self.name is None:
            return
        try:
            self.client.caches.delete(name=self.name)
        except Exception as error:  # noqa: BLE001 - it expires on its own anyway
            logger.info("Could not delete cache %s: %s", self.name, error)
        self.name = None
        self.covered = 0
