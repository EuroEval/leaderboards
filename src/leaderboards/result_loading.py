"""Loading of results, to be converted into leaderboards."""

import json
import logging
import tarfile
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)


def load_raw_results() -> list[dict]:
    """Load raw results.

    Returns:
        The raw results.

    Raises:
        FileNotFoundError:
            If the raw results file is not found.
        ValueError:
            If the raw results file contains invalid JSON.
    """
    results_path = Path("results.tar.gz")
    if not results_path.exists():
        raise FileNotFoundError(f"Results file {results_path} not found.")

    logger.info(f"Loading raw results from {results_path}...")

    # Unpack the tar.gz file in memory and read the JSONL file
    with tarfile.open(results_path, "r:gz") as tar:
        results_file = tar.extractfile(member="results/results.jsonl")
        if results_file is None:
            raise FileNotFoundError(
                "results/results.jsonl not found in the tar.gz file."
            )
        result_lines = results_file.read().decode(encoding="utf-8").splitlines()
        logger.info(f"Loaded {len(result_lines):,} existing results.")

    # If there are new results, add them to the existing results
    new_results_path = Path("new_results.jsonl")
    if new_results_path.exists():
        with new_results_path.open() as f:
            new_result_lines = f.read().splitlines()
        result_lines.extend(new_result_lines)
        new_results_path.unlink()
        logger.info(f"Loaded {len(new_result_lines):,} new results.")

    # Parse each line as JSON, skipping empty lines
    records = list()
    for line_idx, line in enumerate(result_lines):
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


@cache
def load_processed_results() -> list[dict]:
    """Load processed results.

    Returns:
        The processed results.

    Raises:
        FileNotFoundError:
            If the processed results file is not found.
        ValueError:
            If the processed results file contains invalid JSON.
    """
    results_path = Path("results.tar.gz")
    if not results_path.exists():
        raise FileNotFoundError("Processed results file not found.")

    logger.info(f"Loading processed results from {results_path}...")

    # Unpack the tar.gz file in memory and read the JSONL file
    with tarfile.open(results_path, "r:gz") as tar:
        results_file = tar.extractfile(member="results/results.processed.jsonl")
        if results_file is None:
            raise FileNotFoundError(
                "results/results.processed.jsonl not found in the tar.gz file."
            )
        result_lines = results_file.read().decode(encoding="utf-8").splitlines()

    # Parse each line as JSON, skipping empty lines
    results = list()
    for line_idx, line in enumerate(result_lines):
        if not line.strip():
            continue
        for record in line.replace("}{", "}\n{").split("\n"):
            if not record.strip():
                continue
            try:
                results.append(json.loads(record))
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON on line {line_idx:,}: {record}.")

    return results
