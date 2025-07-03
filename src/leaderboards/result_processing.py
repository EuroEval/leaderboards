"""Process EuroEval records from a JSONL file."""

import json
import logging
import re
import warnings
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HFValidationError
from huggingface_hub.hf_api import RepositoryNotFoundError
from tqdm.auto import tqdm

from .cache import Cache
from .link_generation import generate_anchor_tag
from .result_loading import load_raw_results
from .utils import extract_model_id_from_record, get_record_hash

logger = logging.getLogger(__name__)


def process_results(
    records_path: Path,
    banned_versions: list[str],
    banned_model_patterns: list[re.Pattern],
    api_model_patterns: list[re.Pattern],
) -> None:
    """Process EuroEval records from a JSONL file.

    Args:
        records_path:
            The path to the JSONL file containing the EuroEval records.
        banned_versions:
            A list of banned EuroEval versions to filter out.
        banned_model_patterns:
            A list of regex patterns to filter out models that should not be included.
        api_model_patterns:
            A list of regex patterns for API inference models.
    """
    # Build the cache from the processed records file
    processed_records_path = records_path.with_suffix(".processed.jsonl")
    cache = Cache.from_processed_records(processed_records_path=processed_records_path)

    # Load all the raw records
    records = load_raw_results(records_path=records_path)
    num_raw_records = len(records)

    # Remove duplicates from the raw records
    all_hash_values = [get_record_hash(record=dct) for dct in records]
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
        logger.info(f"Removed {num_duplicates:,} duplicates from {records_path}.")

    # Overwrite original scores file with the de-duplicated records, to avoid bloat
    with records_path.open(mode="w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    records = [
        add_missing_entries(record=record, cache=cache)
        for record in tqdm(records, desc="Adding missing entries")
    ]

    records = [
        fixed_record
        for record in tqdm(records, desc="Fixing metadata in records")
        if (fixed_record := fix_metadata(record=record, cache=cache)) is not None
    ]

    # Remove invalid evaluation records
    records = [
        record
        for record in records
        if record_is_valid(
            record=record,
            banned_versions=banned_versions,
            banned_model_patterns=banned_model_patterns,
            api_model_patterns=api_model_patterns,
        )
    ]
    num_invalid_records = num_raw_records - num_duplicates - len(records)
    if num_invalid_records > 0:
        logger.info(
            f"Removed {num_invalid_records:,} invalid records from {records_path}."
        )

    # Store processed records in separate file
    with records_path.with_suffix(".processed.jsonl").open(mode="w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def add_missing_entries(record: dict, cache: Cache) -> dict:
    """Adds missing entries to a record.

    Args:
        record:
            A record from the JSONL file.
        cache:
            The cache.

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
    if "generative_type" not in record:
        record["generative_type"] = get_generative_type(record=record, cache=cache)
    record["merge"] = is_merge(record=record, cache=cache)
    record["commercially_licensed"] = is_commercially_licensed(
        record=record, cache=cache
    )
    return record


def fix_metadata(record: dict, cache: Cache) -> dict | None:
    """Fixes metadata in a record.

    Args:
        record:
            A record from the JSONL file.

    Returns:
        The record with fixed metadata, or None if the record should be removed.
    """
    if record["task"] == "question-answering":
        record["task"] = "reading-comprehension"
    if "scandeval_version" in record:
        record["euroeval_version"] = record["scandeval_version"]
        del record["scandeval_version"]
    if record["model"] in cache.anchor_tag:
        record["model"] = cache.anchor_tag[record["model"]]
    else:
        anchor_tag = generate_anchor_tag(model_id=record["model"])
        if anchor_tag is None:
            return None
        cache.anchor_tag[record["model"]] = anchor_tag
        record["model"] = anchor_tag
    return record


def get_generative_type(record: dict, cache: Cache) -> str | None:
    """Asks for the generative type of a model.

    Args:
        record:
            A record from the JSONL file.
        cache:
            The cache.

    Returns:
        The generative type of the model.
    """
    # Remove revisions from model ID
    model_id = record["model"].split("@")[0]

    while True:
        if model_id in cache.generative_type:
            return cache.generative_type[model_id]

        # Pre-fill based on keywords in model name
        null_keywords = ["bert", "xlm-r", "encoder"]
        base_keywords = ["-base", "-pt"]
        instruct_keywords = ["-instruct", "-it$", "-chat"]
        reasoning_keywords = ["^o[1-9]$", "^o[1-9]-", "deepseek-r1"]
        if any(
            re.search(pattern=keyword, string=model_id, flags=re.IGNORECASE)
            for keyword in null_keywords
        ):
            cache.generative_type[model_id] = None
            return None
        if any(
            re.search(pattern=keyword, string=model_id, flags=re.IGNORECASE)
            for keyword in base_keywords
        ):
            cache.generative_type[model_id] = "base"
            return "base"
        if any(
            re.search(pattern=keyword, string=model_id, flags=re.IGNORECASE)
            for keyword in instruct_keywords
        ):
            cache.generative_type[model_id] = "instruction_tuned"
            return "instruction_tuned"
        if any(
            re.search(pattern=keyword, string=model_id, flags=re.IGNORECASE)
            for keyword in reasoning_keywords
        ):
            cache.generative_type[model_id] = "reasoning"
            return "reasoning"

        msg = f"What is the generative type of {model_id!r}?"
        if "/" in model_id:
            msg += f" (https://hf.co/{model_id})"
        msg += " [0=null, 1=base, 2=instruction_tuned, 3=reasoning] "
        user_input = input(msg)
        if user_input.lower() in {"0", "null"}:
            cache.generative_type[model_id] = None
        elif user_input.lower() in {"1", "base"}:
            cache.generative_type[model_id] = "base"
        elif user_input.lower() in {"2", "instruction_tuned"}:
            cache.generative_type[model_id] = "instruction_tuned"
        elif user_input.lower() in {"3", "reasoning"}:
            cache.generative_type[model_id] = "reasoning"
        else:
            print("Invalid input. Please try again.")


def is_commercially_licensed(record: dict, cache: Cache) -> bool:
    """Asks if a model is commercially licensed.

    Args:
        record:
            A record from the JSONL file.
        cache:
            The cache.

    Returns:
        Whether the model is commercially licensed.
    """
    # Remove revisions from model ID
    model_id = record["model"].split("@")[0]

    # Assume that non-generative models are always commercially licensed
    if not record.get("generative", True):
        cache.commercially_licensed[model_id] = True

    while True:
        if model_id in cache.commercially_licensed:
            return cache.commercially_licensed[model_id]

        msg = f"Is {model_id!r} commercially licensed?"
        if "/" in model_id:
            msg += f" (https://hf.co/{model_id})"
        msg += " [y/n] "
        user_input = input(msg)
        if user_input.lower() in {"y", "yes"}:
            cache.commercially_licensed[model_id] = True
        elif user_input.lower() in {"n", "no"}:
            cache.commercially_licensed[model_id] = False
        else:
            logger.error("Invalid input. Please try again.")
            continue


def is_merge(record: dict, cache: Cache) -> bool:
    """Determines if a model is a merged model.

    Args:
        record:
            A record from the JSONL file.
        cache:
            The cache.

    Returns:
        Whether the model is a merged model.
    """
    # Remove revisions from model ID
    model_id = record["model"].split("@")[0]

    # Return cached value if available
    if model_id in cache.merge:
        return cache.merge[model_id]

    # Fresh models do not appear on the model hub, so we assume they are not merge
    # models
    if model_id.startswith("fresh"):
        cache.merge[model_id] = False
        return False

    # Fetch model info from the model hub, and assume that it is not a merged model if
    # the model is not found
    api = HfApi()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model_info = api.model_info(repo_id=model_id)
    except (RepositoryNotFoundError, HFValidationError):
        cache.merge[model_id] = False
        return False

    # A model is a merge model if it has merge-related tags
    merge_tags = ["merge", "mergekit"]
    has_merge_tag = any(tag in (model_info.tags or []) for tag in merge_tags)
    cache.merge[model_id] = has_merge_tag
    return has_merge_tag


def record_is_valid(
    record: dict,
    banned_versions: list[str],
    banned_model_patterns: list[re.Pattern],
    api_model_patterns: list[re.Pattern],
) -> bool:
    """Determine if a record is valid.

    Args:
        record:
            The record to validate.

    Returns:
        True if the record is valid, False otherwise.
    """
    # Remove records with banned EuroEval versions
    if record.get("euroeval_version") in banned_versions:
        return False

    # Remove banned models
    if any(
        re.search(pattern=pattern, string=record["model"])
        for pattern in banned_model_patterns
    ):
        return False

    # Do not allow few-shot evaluation for API models
    model_id_match = re.search(r">(.+?)<", record["model"])
    model_id = model_id_match.group(1) if model_id_match else record["model"]
    if any(
        re.fullmatch(pattern=pattern, string=model_id) for pattern in api_model_patterns
    ) and record.get("few_shot", True):
        return False

    # Otherwise, the record is valid
    return True


def group_results_by_model(
    results: list[dict], task_config: dict[str, dict[str, str]]
) -> dict[str, dict[str, list[tuple[list[float], float, float]]]]:
    """Group results by model ID.

    Args:
        results:
            The processed results.
        task_config:
            The task configuration.

    Returns:
        The results grouped by model ID.
    """
    model_scores: dict[str, dict[str, list[tuple[list[float], float, float]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for record in results:
        model_id = extract_model_id_from_record(record=record)
        dataset: str = record["dataset"]

        for metric_type in ["primary", "secondary"]:
            metric = task_config[record["task"]][f"{metric_type}_metric"]

            # Get the metrics for the dataset
            if "test" in record["results"]["raw"]:
                raw_scores = [
                    result_dict.get(f"test_{metric}", result_dict.get(metric, -1))
                    for result_dict in record["results"]["raw"]["test"]
                ]
            else:
                raw_scores = [
                    result_dict.get(f"test_{metric}", result_dict.get(metric, -1))
                    for result_dict in record["results"]["raw"]
                ]

            # Get the aggregated scores for the dataset
            total_score: float = record["results"]["total"][f"test_{metric}"]
            std_err: float = record["results"]["total"][f"test_{metric}_se"]

            # Sometimes the raw scores are normalised to [0, 1], so we need to scale
            # them back to [0, 100]
            if max(raw_scores) <= 1:
                raw_scores = [score * 100 for score in raw_scores]

            model_scores[model_id][dataset].append((raw_scores, total_score, std_err))

    return model_scores


def extract_model_metadata(results: list[dict]) -> dict[str, dict]:
    """Extract metadata from the results.

    Args:
        results:
            The processed results.

    Returns:
        The metadata.
    """
    metadata_dict: dict[str, dict] = defaultdict(dict)
    for record in results:
        model_id = extract_model_id_from_record(record=record)
        num_params = (
            record["num_model_parameters"]
            if record["num_model_parameters"] >= 0
            else float("nan")
        )
        vocab_size = (
            record["vocabulary_size"]
            if record["vocabulary_size"] >= 0
            else float("nan")
        )
        context = (
            record["max_sequence_length"]
            if record["max_sequence_length"] >= 0
            else float("nan")
        )
        metadata_dict[model_id].update(
            dict(
                parameters=num_params,
                vocabulary_size=vocab_size,
                context=context,
                generative_type=record.get("generative_type", None),
                commercial=record.get("commercially_licensed", False),
                merge=record.get("merge", False),
            )
        )

        version = record.get("euroeval_version", "<9.2.0")
        if version != "<9.2.0":
            version_sort_value = int(
                "".join(
                    [
                        f"{version_part:0>2}"
                        for version_part in re.sub(
                            pattern=r"\.dev[0-9]+", repl="", string=version
                        ).split(".")
                    ]
                )
            )
            version += f"@@{version_sort_value}"
        else:
            version += "@@0"
        metadata_dict[model_id][f"{record['dataset']}_version"] = version

    return metadata_dict
