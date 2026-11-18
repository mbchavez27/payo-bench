"""Load versioned prompt files and fill their {{placeholders}}."""

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


@dataclass(frozen=True)
class PromptTemplate:
    """A prompt file: its body (the only part sent to a model) plus identifying metadata."""

    name: str
    version: str
    body: str
    sha256: str
    placeholders: frozenset[str]


def load_prompt(path: Path) -> PromptTemplate:
    """
    Read a prompt file and split off its YAML front-matter.

    Args:
        path: Path to a `<task>_v<N>.md` prompt file.

    Returns:
        The parsed template; `sha256` is of the whole file, so any edit changes it.
    """
    text = path.read_text(encoding="utf-8")
    match = FRONT_MATTER.match(text)
    if match is None:
        raise ValueError(f"{path.name}: missing front-matter")
    version = re.search(r"^version:\s*(\S+)", match.group(1), re.M)
    if version is None:
        raise ValueError(f"{path.name}: front-matter has no version")
    body = text[match.end():]
    return PromptTemplate(
        name=path.stem,
        version=version.group(1),
        body=body,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        placeholders=frozenset(PLACEHOLDER.findall(body)),
    )


def render(template: PromptTemplate, values: Mapping[str, str]) -> str:
    """
    Fill every placeholder by plain string replacement.

    Args:
        template: The loaded prompt.
        values: One value per placeholder in the body; extra keys are ignored.

    Returns:
        The prompt text to send as a single user turn.
    """
    missing = template.placeholders - values.keys()
    if missing:
        raise KeyError(f"{template.name}: no value for {sorted(missing)}")
    text = template.body
    for key in template.placeholders:
        text = text.replace("{{" + key + "}}", values[key])
    leftover = PLACEHOLDER.findall(text)
    if leftover:
        raise ValueError(f"{template.name}: placeholders left after rendering: {leftover}")
    return text


def format_post(problem_goal: str, context: str, previous_attempts: str | None) -> str:
    """
    Format a post's template fields the way every prompt shows them (flair is never included).

    Args:
        problem_goal: Text under the Problem/Goal header.
        context: Text under the Context header.
        previous_attempts: Text under Previous Attempts, or None when the post has none.

    Returns:
        The `{{post}}` block.
    """
    parts = [f"Problem/Goal: {problem_goal.strip()}", f"Context: {context.strip()}"]
    if previous_attempts and previous_attempts.strip():
        parts.append(f"Previous Attempts: {previous_attempts.strip()}")
    return "\n\n".join(parts)


def sha256_text(text: str) -> str:
    """Return the sha256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
