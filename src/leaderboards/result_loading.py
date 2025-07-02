"""Loading of results, to be converted into leaderboards."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_raw_results(records_path: Path) -> list[dict]:
    """Load raw results.

    Returns:
        The raw results.

    Raises:
        FileNotFoundError:
            If the raw results file is not found.
        ValueError:
            If the raw results file contains invalid JSON.
    """
    if not records_path.exists():
        raise FileNotFoundError(f"Raw results file {records_path} not found.")

    logger.info(f"Loading raw results from {records_path}...")

    records = list()
    with records_path.open() as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue

            # We split on '}{' to handle cases where multiple JSON objects are on the
            # same line
            for record in line.replace("}{", "}\n{").split("\n"):
                if not record.strip():
                    continue
                try:
                    records.append(json.loads(record))
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON on line {line_idx:,}: {record}.")

    return records


def load_processed_results(allowed_datasets: list[str]) -> list[dict]:
    """Load processed results.

    Args:
        allowed_datasets:
            The list of datasets to include in the leaderboard.

    Returns:
        The processed results.

    Raises:
        FileNotFoundError:
            If the processed results file is not found.
    """
    results_path = Path("results", "results.processed.jsonl")
    if not results_path.exists():
        raise FileNotFoundError("Processed results file not found.")

    logger.info(f"Loading processed results from {results_path}...")

    results = list()
    with results_path.open() as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            for record in line.replace("}{", "}\n{").split("\n"):
                if not record.strip():
                    continue
                try:
                    results.append(json.loads(record))
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON on line {line_idx:,}: {record}.")

    # Only keep relevant results
    results = [record for record in results if record["dataset"] in allowed_datasets]

    return results
