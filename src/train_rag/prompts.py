"""Load the prompts from `prompts.yaml`, the one place every LLM prompt is kept."""

import re
from functools import cache
from pathlib import Path

import yaml

PROMPTS_FILE = Path(__file__).with_name("prompts.yaml")
# A line break that isn't part of a blank line: the file wraps long lines for reading.
WRAPPED_LINE = re.compile(r"(?<!\n)\n(?!\n)")


@cache
def load_prompts() -> dict[str, str]:
    """Read every prompt from the file, once, joining wrapped lines into paragraphs.

    Returns:
        Prompt texts keyed by name.
    """
    prompts = yaml.safe_load(PROMPTS_FILE.read_text())

    return {name: WRAPPED_LINE.sub(" ", str(text).strip()) for name, text in prompts.items()}


def prompt(name: str, **values: object) -> str:
    """Get one prompt with its placeholders filled in.

    Args:
        name: Key of the prompt in `prompts.yaml`.
        **values: Values for the prompt's `{placeholders}`.

    Returns:
        The prompt text.

    Raises:
        KeyError: If there is no prompt with this name, or a placeholder has no value.
    """
    return load_prompts()[name].format(**values)
