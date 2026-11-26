"""The label schema: the single source of label ids and one-line definitions."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from payo_llm import ROOT

SCHEMA_PATH = ROOT / "02-llm" / "gold" / "emotion-recognition" / "label-schema.json"


@dataclass(frozen=True)
class Schema:
    """Read-only view of label-schema.json."""

    raw: dict[str, Any]

    @property
    def version(self) -> str:
        return self.raw["version"]

    @property
    def reply_language(self) -> str:
        return self.raw["reply_language"]

    @property
    def emotion_ids(self) -> frozenset[str]:
        return frozenset(item["id"] for item in self.raw["emotions"])

    @property
    def concept_ids(self) -> frozenset[str]:
        return frozenset(item["id"] for item in self.raw["cultural_concepts"])

    @property
    def concept_aliases(self) -> dict[str, str]:
        """Emotion ids accepted in the concept field, mapped to their concept id (fixed, logged rule)."""
        aliases = dict(self.raw.get("concept_aliases") or self.raw.get("legacy_id_map", {}).get("cultural_concept", {}))
        bad = {k: v for k, v in aliases.items() if v not in self.concept_ids}
        if bad:
            raise ValueError(f"concept_aliases point to unknown concept ids: {bad}")
        return aliases

    @property
    def legacy_id_map(self) -> dict[str, Any]:
        """Map for migrating v1 and pilot labels."""
        return dict(self.raw.get("legacy_id_map", {}))

    def render_list(self, key: str) -> str:
        """
        Render one schema list for a prompt, in schema order.

        Args:
            key: A list in the schema, e.g. "emotions" or "cultural_concepts".

        Returns:
            One line per item: `- <id>: <definition>`.
        """
        return "\n".join(f"- {item['id']}: {item['definition']}" for item in self.raw[key])

    def emotion(self, emotion_id: str) -> dict[str, str]:
        """Return the schema entry for an emotion id."""
        return self._find("emotions", emotion_id)

    def concept(self, concept_id: str) -> dict[str, str]:
        """Return the schema entry for a cultural-concept id."""
        return self._find("cultural_concepts", concept_id)

    def _find(self, key: str, item_id: str) -> dict[str, str]:
        for item in self.raw[key]:
            if item["id"] == item_id:
                return item
        raise KeyError(f"{item_id!r} is not in schema list {key!r}")


def load_schema(path: Path = SCHEMA_PATH) -> Schema:
    """
    Load the label schema.

    Args:
        path: Path to label-schema.json.

    Returns:
        The schema.
    """
    return Schema(json.loads(path.read_text(encoding="utf-8")))
