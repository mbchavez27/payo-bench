"""Runners for the two LLM tasks in payo-bench: emotion recognition and advice generation."""

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """
    Find the repository root by walking up to the folder that holds pyproject.toml.

    Args:
        start: Folder to start from; defaults to this file's folder.

    Returns:
        Absolute path of the repository root.
    """
    here = (start or Path(__file__)).resolve()
    for folder in [here, *here.parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    raise FileNotFoundError("pyproject.toml not found above " + str(here))


ROOT = repo_root()
