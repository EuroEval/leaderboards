"""Loading of results, to be converted into leaderboards."""

import json
import logging
import re
import tarfile
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_to_old_format(record: dict) -> dict:
    """Convert a record from the new Every Eval format to the old EuroEval format.

    The new format (schema_version 0.2.1) has:
    - evaluation_results: list of evaluation results with score_details.score
    - eval_library.additional_details.raw_results: JSON string of raw per-fold results

    The old format has:
    - results.raw: list of raw score dicts
    - results.total: dict with aggregated scores

    Args:
        record: A record from the JSONL file.

    Returns:
        The record with results converted to the old format.
    """
    schema_version = record.get("schema_version", "0.0.0")

    # If the record is already in the old format (no schema_version or < 0.2.0),
    # return it unchanged
    version_parts = list(
        map(
            int,
            re.sub(pattern=r"\.dev[0-9]+", repl="", string=schema_version).split("."),
        )
    )
    if version_parts[0] < 0 or (version_parts[0] == 0 and version_parts[1] < 2):
        return record

    # Convert from new format to old format
    new_record: dict = dict(record)
    new_record["results"] = {"raw": {}, "total": {}}

    # Extract raw results from eval_library.additional_details.raw_results
    raw_results_str = (
        record.get("eval_library", {}).get("additional_details", {}).get("raw_results")
    )
    if raw_results_str:
        raw_results = json.loads(raw_results_str)
        # Use "test" as the split name for consistency with old format
        new_record["results"]["raw"]["test"] = raw_results

    # Extract evaluation results and convert to old format
    # The evaluation_results contains entries like test_mcc, test_accuracy, etc.
    for eval_result in record.get("evaluation_results", []):
        evaluation_name = eval_result.get("evaluation_name", "")
        score = eval_result.get("score_details", {}).get("score", 0)

        # Convert metric name to match old format (e.g., "test_mcc" -> "test_mcc")
        # and add standard error (bootstrap CI width / 3.92 for 95% CI)
        if evaluation_name.startswith("test_"):
            metric_name = evaluation_name
        else:
            metric_name = f"test_{evaluation_name}"

        new_record["results"]["total"][metric_name] = score

        # Calculate standard error from confidence interval
        uncertainty = eval_result.get("score_details", {}).get("uncertainty", {})
        ci = uncertainty.get("confidence_interval", {})
        if ci.get("lower") is not None and ci.get("upper") is not None:
            # For 95% CI, width = 3.92 * SE, so SE = width / 3.92
            ci_width = ci["upper"] - ci["lower"]
            se = ci_width / 3.92
            new_record["results"]["total"][f"{metric_name}_se"] = se

    return new_record


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
        # Convert new results from new format to old format
        converted_new_records = [
            convert_to_old_format(record=json.loads(line))
            for line in new_result_lines
            if line.strip()
        ]
        result_lines.extend(json.dumps(record) for record in converted_new_records)
        new_results_path.unlink()
        logger.info(f"Loaded {len(converted_new_records):,} new results.")

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
                parsed_record = json.loads(record)
                # Convert from new Every Eval format to old EuroEval format
                converted_record = convert_to_old_format(record=parsed_record)
                records.append(converted_record)
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
