"""Caching of model metadata."""

import json
import logging
import re
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Cache:
    """A cache for model metadata.

    Attributes:
        generative_type:
            A mapping from model IDs to their generative type.
        merge:
            A mapping from model IDs to whether they are merges of other models.
        commercially_licensed:
            A mapping from model IDs to whether they are commercially licensed.
        anchor_tag:
            A mapping from model IDs to their anchor tag.
    """

    generative_type: dict[str, str | None] = field(default_factory=dict)
    merge: dict[str, bool] = field(default_factory=dict)
    commercially_licensed: dict[str, bool] = field(default_factory=dict)
    anchor_tag: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_processed_records(cls, processed_records_path: Path) -> "Cache":
        """Create a cache from processed records.

        Args:
            processed_records_path:
                The path to the processed records file.

        Returns:
            A Cache instance populated with model metadata.

        Raises:
            ValueError:
                If the processed records file contains invalid JSON.
        """
        # If the processed records file does not exist, return empty cache
        if not processed_records_path.exists():
            logger.warning(
                f"Processed records file {processed_records_path} does not exist. "
                "Using an empty cache."
            )
            return cls()

        # Load the processed records
        old_records: list[dict[str, t.Any]] = list()
        with processed_records_path.open() as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                for line in line.replace("}{", "}\n{").split("\n"):
                    if not line.strip():
                        continue
                    try:
                        old_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON on line {line_idx:,}: {line}.")

        # Populate a cache from the old records
        cache = cls()
        for record in tqdm(old_records, desc="Building caches"):
            model_id: str = record["model"]
            if (match := re.search(r">(.+?)<", record["model"])) is not None:
                model_id = match.group(1)
            model_id = model_id.split("@")[0]
            if "generative_type" in record:
                cache.generative_type[model_id] = record["generative_type"]
            if "merge" in record:
                cache.merge[model_id] = record["merge"]
            if "commercially_licensed" in record:
                cache.commercially_licensed[model_id] = record["commercially_licensed"]
            if record["model"].startswith("<a href="):
                inner_model_id_match = re.search(r">(.+?)<", record["model"])
                if inner_model_id_match:
                    inner_model_id = inner_model_id_match.group(1)
                    inner_model_id = re.sub(r" *\(.*?\)", "", inner_model_id)
                    cache.anchor_tag[inner_model_id] = record["model"]

        return cache
