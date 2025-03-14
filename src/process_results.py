"""Process EuroEval records from a JSONL file."""

import warnings
import click
import json
from pathlib import Path
import logging
from huggingface_hub import HfApi
from huggingface_hub.errors import HFValidationError
from huggingface_hub.hf_api import RepositoryNotFoundError
from tqdm.auto import tqdm
import typing as t
import re

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


BANNED_VERSIONS: list[str] = ["9.3.0", "10.0.0"]
BANNED_MODELS: list[re.Pattern] = []
GENERATIVE_TYPE_CACHE: dict[str, str | None] = dict()
MERGE_CACHE: dict[str, bool] = dict()
COMMERCIALLY_LICENSED_CACHE: dict[str, bool] = dict()


@click.command()
@click.argument("filename")
def main(filename: str) -> None:
    """Process EuroEval records from a JSONL file.

    Args:
        filename:
            The path to the JSONL file.
    """
    # Build caches
    global GENERATIVE_TYPE_CACHE
    global MERGE_CACHE
    global COMMERCIALLY_LICENSED_CACHE
    old_records: list[dict[str, t.Any]] = list()
    with Path(filename).with_suffix(".processed.jsonl").open(mode="r") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            for line in line.replace("}{", "}\n{").split("\n"):
                if not line.strip():
                    continue
                try:
                    old_records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON on line {line_idx:,}: {line}.")
                    return
    for record in tqdm(old_records, desc="Building caches"):
        model_id: str = record["model"]
        model_id = model_id.split("@")[0]
        if "generative_type" in record:
            GENERATIVE_TYPE_CACHE[model_id] = record["generative_type"]
        if "merge" in record:
            MERGE_CACHE[model_id] = record["merge"]
        if "commercially_licensed" in record:
            COMMERCIALLY_LICENSED_CACHE[model_id] = record["commercially_licensed"]
    del old_records

    records = list()
    with Path(filename).open(mode="r") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            for record in line.replace("}{", "}\n{").split("\n"):
                if not record.strip():
                    continue
                try:
                    records.append(json.loads(record))
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON on line {line_idx:,}: {record}.")
                    return
    num_raw_records = len(records)

    # Remove duplicates
    all_hash_values = [get_hash(dct) for dct in records]
    unique_hash_values = sorted(set(all_hash_values))
    new_records = list()
    for unique_hash_value in tqdm(unique_hash_values, desc="Processing records"):
        matches = [
            record
            for record, hash_value in zip(records, all_hash_values)
            if hash_value == unique_hash_value
        ]
        versions = [
            list(
                map(
                    int,
                    re.sub(
                        pattern=r"\.dev[0-9]+",
                        repl="",
                        string=match.get(
                            "euroeval_version", match.get("scandeval_version", "0.0.0")
                        ),
                    ).split("."),
                )
            )
            for match in matches
        ]
        newest_version = max(versions)
        matches_with_newest_version = [
            match
            for match, version in zip(matches, versions)
            if version == newest_version
        ]
        newest_match = matches_with_newest_version[-1]
        new_records.append(newest_match)
    records = new_records
    num_duplicates = num_raw_records - len(records)
    if num_duplicates:
        logger.info(f"Removed {num_duplicates:,} duplicates from {filename}.")

    # Overwrite original scores file with the de-duplicated records
    with Path(filename).open(mode="w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    records = [
        add_missing_entries(record=record)
        for record in tqdm(records, desc="Adding missing entries")
    ]

    records = [
        fix_metadata(record=record)
        for record in tqdm(records, desc="Fixing metadata in records")
    ]

    # Remove invalid evaluation records
    records = [record for record in records if record_is_valid(record=record)]
    num_invalid_records = num_raw_records - num_duplicates - len(records)
    if num_invalid_records > 0:
        logger.info(f"Removed {num_invalid_records:,} invalid records from {filename}.")

    # Store processed records in separate file
    with Path(filename).with_suffix(".processed.jsonl").open(mode="w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def add_missing_entries(record: dict) -> dict:
    """Adds missing entries to a record.

    Args:
        record:
            A record from the JSONL file.

    Returns:
        The record with missing entries added.
    """
    if "validation_split" not in record:
        record["validation_split"] = False
    if "few_shot" not in record:
        record["few_shot"] = True
    if "generative" not in record:
        record["generative"] = False
    if "euroeval_version" not in record and "scandeval_version" not in record:
        record["euroeval_version"] = "<9.2.0"
    record["generative_type"] = get_generative_type(record=record)
    record["merge"] = is_merge(record=record)
    record["commercially_licensed"] = is_commercially_licensed(record=record)
    return record


def fix_metadata(record: dict) -> dict:
    """Fixes metadata in a record.

    Args:
        record:
            A record from the JSONL file.

    Returns:
        The record with fixed metadata.
    """
    if record["task"] == "question-answering":
        record["task"] = "reading-comprehension"
    if "scandeval_version" in record:
        record["euroeval_version"] = record["scandeval_version"]
        del record["scandeval_version"]
    return record


def get_generative_type(record: dict) -> str | None:
    """Asks for the generative type of a model.

    Args:
        record:
            A record from the JSONL file.

    Returns:
        The generative type of the model.
    """
    global GENERATIVE_TYPE_CACHE

    # Remove revisions from model ID
    model_id = record["model"].split("@")[0]

    while True:
        if model_id in GENERATIVE_TYPE_CACHE:
            return GENERATIVE_TYPE_CACHE[model_id]

        # Pre-fill based on keywords in model name
        null_keywords = ["bert", "xlm-r"]
        instruct_keywords = ["-instruct", "-it$", "-chat"]
        reasoning_keywords = ["^o[1-9]$", "^o[1-9]-", "deepseek-r1"]
        if any(
            re.search(pattern=keyword, string=model_id, flags=re.IGNORECASE)
            for keyword in null_keywords
        ):
            GENERATIVE_TYPE_CACHE[model_id] = None
            return None
        if any(
            re.search(pattern=keyword, string=model_id, flags=re.IGNORECASE)
            for keyword in instruct_keywords
        ):
            GENERATIVE_TYPE_CACHE[model_id] = "instruction_tuned"
            return "instruction_tuned"
        if any(
            re.search(pattern=keyword, string=model_id, flags=re.IGNORECASE)
            for keyword in reasoning_keywords
        ):
            GENERATIVE_TYPE_CACHE[model_id] = "reasoning"
            return "reasoning"

        msg = f"What is the generative type of {model_id!r}?"
        if "/" in model_id:
            msg += f" (https://huggingface.co/{model_id})"
        msg += " [0=null, 1=base, 2=instruction_tuned, 3=reasoning] "
        user_input = input(msg)
        if user_input.lower() in {"0", "null"}:
            GENERATIVE_TYPE_CACHE[model_id] = None
        elif user_input.lower() in {"1", "base"}:
            GENERATIVE_TYPE_CACHE[model_id] = "base"
        elif user_input.lower() in {"2", "instruction_tuned"}:
            GENERATIVE_TYPE_CACHE[model_id] = "instruction_tuned"
        elif user_input.lower() in {"3", "reasoning"}:
            GENERATIVE_TYPE_CACHE[model_id] = "reasoning"
        else:
            print("Invalid input. Please try again.")


def is_commercially_licensed(record: dict) -> bool:
    """Asks if a model is commercially licensed.

    Args:
        record:
            A record from the JSONL file.

    Returns:
        Whether the model is commercially licensed.
    """
    global COMMERCIALLY_LICENSED_CACHE

    # Remove revisions from model ID
    model_id = record["model"].split("@")[0]

    # Assume that non-generative models are always commercially licensed
    if not record.get("generative", True):
        COMMERCIALLY_LICENSED_CACHE[model_id] = True

    while True:
        if model_id in COMMERCIALLY_LICENSED_CACHE:
            return COMMERCIALLY_LICENSED_CACHE[model_id]

        msg = f"Is {model_id!r} commercially licensed?"
        if "/" in model_id:
            msg += f" (https://huggingface.co/{model_id})"
        msg += " [y/n] "
        user_input = input(msg)
        if user_input.lower() in {"y", "yes"}:
            COMMERCIALLY_LICENSED_CACHE[model_id] = True
        elif user_input.lower() in {"n", "no"}:
            COMMERCIALLY_LICENSED_CACHE[model_id] = False
        else:
            print("Invalid input. Please try again.")
            continue


def get_hash(record: dict) -> str:
    """Returns a hash value for a record.

    Args:
        record:
            A record from the JSONL file.

    Returns:
        A hash value for the record.
    """
    model = record["model"]
    dataset = record["dataset"]
    validation_split = int(record.get("validation_split", False))
    few_shot = int(record.get("few_shot", True))
    generative = int(record.get("generative", False))
    return f"{model}{dataset}{validation_split}{generative * (few_shot + 1)}"


def is_merge(record: dict) -> bool:
    """Determines if a model is a merged model.

    Args:
        record:
            A record from the JSONL file.

    Returns:
        Whether the model is a merged model.
    """
    # Remove revisions from model ID
    model_id = record["model"].split("@")[0]

    # Return cached value if available
    global MERGE_CACHE
    if model_id in MERGE_CACHE:
        return MERGE_CACHE[model_id]

    # Fresh models do not appear on the model hub, so we assume they are not merge
    # models
    if model_id.startswith("fresh"):
        MERGE_CACHE[model_id] = False
        return False

    # Fetch model info from the model hub, and assume that it is not a merged model if
    # the model is not found
    api = HfApi()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model_info = api.model_info(repo_id=model_id)
    except (RepositoryNotFoundError, HFValidationError):
        MERGE_CACHE[model_id] = False
        return False

    # A model is a merge model if it has merge-related tags
    merge_tags = ["merge", "mergekit"]
    has_merge_tag = any(tag in model_info.tags for tag in merge_tags)
    MERGE_CACHE[model_id] = has_merge_tag
    return has_merge_tag


def record_is_valid(record: dict) -> bool:
    """Determine if a record is valid.

    Args:
        record:
            The record to validate.

    Returns:
        True if the record is valid, False otherwise.
    """
    if record.get("euroeval_version") in BANNED_VERSIONS:
        return False
    if any(
        re.search(pattern=pattern, string=record["model"]) for pattern in BANNED_MODELS
    ):
        return False
    return True


if __name__ == "__main__":
    main()
