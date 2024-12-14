"""Generate leaderboard CSV files from the ScandEval results."""

from collections import defaultdict
import json
import math
from pathlib import Path
import warnings
import click
from yaml import safe_load
import logging
import pandas as pd
import scipy.stats as stats
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@click.command()
@click.argument("leaderboard_config")
def main(leaderboard_config: str | Path) -> None:
    """Generate leaderboard CSV files from the ScandEval results.

    Args:
        leaderboard_config:
            The path to the leaderboard configuration file.
    """
    leaderboard_config = Path(leaderboard_config)

    logger.info(f"Generating {leaderboard_config.stem.title()} leaderboard...")

    # Load configs
    with Path("task_config.yaml").open(mode="r") as f:
        task_config: dict[str, dict[str, str]] = safe_load(stream=f)
    with leaderboard_config.open(mode="r") as f:
        config: dict[str, list[str]] = safe_load(stream=f)

    # Load the list of all datasets to be used in the leaderboard
    datasets = [
        dataset for task_datasets in config.values() for dataset in task_datasets
    ] + ["speed"]

    required_datasets_per_category = list()
    categories = {task_config[task]["category"] for task in config}
    for category in categories:
        category_tasks = [
            task for task in config if task_config[task]["category"] == category
        ]
        category_datasets = [
            dataset for task in category_tasks for dataset in config[task]
        ] + ["speed"]
        required_datasets_per_category.append(category_datasets)

    # Load results and set them up for the leaderboard
    results = load_results(allowed_datasets=datasets)
    model_results = group_results_by_model(
        results=results,
        task_config=task_config,
        required_datasets_per_category=required_datasets_per_category,
    )
    ranks = compute_ranks(
        model_results=model_results, task_config=task_config, config=config
    )
    metadata_dict = extract_model_metadata(results=results)

    # Generate the leaderboard and store it to disk
    leaderboard_path = Path("leaderboards") / f"{leaderboard_config.stem}.csv"
    df = generate_dataframe(
        model_results=model_results,
        ranks=ranks,
        metadata_dict=metadata_dict,
        datasets=datasets,
    )
    df.to_csv(leaderboard_path, index=False)


def load_results(allowed_datasets: list[str]) -> list[dict]:
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
    results_path = Path("results/results.processed.jsonl")
    if not results_path.exists():
        raise FileNotFoundError("Processed results file not found.")

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


def group_results_by_model(
    results: list[dict],
    task_config: dict[str, dict[str, str]],
    required_datasets_per_category: list[list[str]],
) -> dict[str, dict[str, tuple[list[float], float]]]:
    """Group results by model ID.

    Args:
        results:
            The processed results.
        task_config:
            The task configuration.
        required_datasets_per_category:
            A list of required datasets per task category. For a model to be included
            in the leaderboard, it needs to have scores for all datasets in at least one
            of the task categories.

    Returns:
        The results grouped by model ID.
    """
    model_scores: dict[str, dict[str, tuple[list[float], float]]] = defaultdict(dict)
    for record in results:
        model_id = extract_model_id_from_record(record=record)
        dataset: str = record["dataset"]
        metric = task_config[record["task"]]["metric"]

        # Get the metrics for the dataset
        total_score: float = record["results"]["total"][f"test_{metric}"]
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

        # Sometimes the raw scores are normalised to [0, 1], so we need to scale them
        # back to [0, 100]
        if max(raw_scores) <= 1:
            raw_scores = [score * 100 for score in raw_scores]

        model_scores[model_id][dataset] = (raw_scores, total_score)

    # Remove the models that don't have scores for all datasets in at least one category
    model_scores = {
        model_id: scores
        for model_id, scores in model_scores.items()
        if any(
            all(dataset in scores for dataset in datasets)
            for datasets in required_datasets_per_category
        )
    }

    return model_scores


def compute_ranks(
    model_results: dict[str, dict[str, tuple[list[float], float]]],
    task_config: dict[str, dict[str, str]],
    config: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """Compute the ranks of the models.

    Args:
        model_results:
            The model results.
        task_config:
            The task configuration.
        config:
            The leaderboard configuration.

    Returns:
        The ranks of the models, per task category.
    """
    datasets = [
        dataset for task_datasets in config.values() for dataset in task_datasets
    ]

    model_dataset_ranks: dict[str, dict[str, float]] = defaultdict(dict)
    for dataset in datasets:
        dummy_scores: tuple[list[float], float] = ([], float("nan"))
        model_dataset_scores = sorted(
            [
                (
                    model_id,
                    scores.get(dataset, dummy_scores)[0],
                    scores.get(dataset, dummy_scores)[1],
                )
                for model_id, scores in model_results.items()
            ],
            key=lambda x: x[2],
            reverse=True,
        )
        stddev = np.std(
            [score for _, _, score in model_dataset_scores if not np.isnan(score)]
        )

        rank_score = 1.0
        previous_scores: list[float] = list()
        for model_id, raw_scores, _ in model_dataset_scores:
            if raw_scores == []:
                model_dataset_ranks[model_id][dataset] = math.inf
                continue
            elif previous_scores == []:
                previous_scores = raw_scores
            elif significantly_better(previous_scores, raw_scores):
                difference = np.mean(previous_scores) - np.mean(raw_scores)
                normalised_difference = difference / stddev
                rank_score += normalised_difference.item()
                previous_scores = raw_scores
            model_dataset_ranks[model_id][dataset] = rank_score

    model_task_ranks: dict[str, dict[str, float]] = defaultdict(dict)
    for model_id, dataset_ranks in model_dataset_ranks.items():
        for task, datasets in config.items():
            model_task_ranks[model_id][task] = np.mean(
                [
                    dataset_ranks[dataset]
                    for dataset in datasets
                    if dataset in dataset_ranks
                ]
            ).item()

    categories = {task_config[task]["category"] for task in config}
    model_task_category_ranks: dict[str, dict[str, float]] = defaultdict(dict)
    for model_id, task_scores in model_task_ranks.items():
        for category in categories:
            model_task_category_ranks[model_id][category] = np.mean(
                [
                    task_scores[task]
                    for task in config
                    if task_config[task]["category"] == category
                ]
            ).item()

    return model_task_category_ranks


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
            round(record["num_model_parameters"] / 1_000_000)
            if record["num_model_parameters"] >= 0
            else "N/A"
        )
        vocab_size = (
            round(record["vocabulary_size"] / 1_000)
            if record["vocabulary_size"] >= 0
            else "N/A"
        )
        context = (
            record["max_sequence_length"]
            if record["max_sequence_length"] >= 0
            else "N/A"
        )
        metadata_dict[model_id].update(
            dict(
                parameters=num_params,
                vocabulary_size=vocab_size,
                context=context,
                commercial=record.get("commercially_licensed", False),
                merge=record.get("merge", False),
            )
        )
        if record["dataset"] == "speed":
            metadata_dict[model_id]["speed"] = record["results"]["total"]["test_speed"]

        metadata_dict[model_id][f"{record['dataset']}_version"] = record.get(
            "scandeval_version", "0.0.0"
        )

    return metadata_dict


def extract_model_id_from_record(record: dict) -> str:
    """Extract the model ID from a record.

    Args:
        record:
            The record.

    Returns:
        The model ID.
    """
    model_id: str = record["model"]
    model_notes: list[str] = list()

    if record.get("generative", True):
        if record.get("few_shot", True):
            model_notes.append("few-shot")
        else:
            model_notes.append("zero-shot")

    if record.get("validation_split", False):
        model_notes.append("val")

    if model_notes:
        model_id += f" ({', '.join(model_notes)})"

    return model_id


def generate_dataframe(
    model_results: dict[str, dict[str, tuple[list[float], float]]],
    ranks: dict[str, dict[str, float]],
    metadata_dict: dict[str, dict],
    datasets: list[str],
) -> pd.DataFrame:
    """Generate a DataFrame from the model results.

    Args:
        model_results:
            The model results.
        ranks:
            The ranks of the models.
        metadata_dict:
            The metadata.
        datasets:
            All datasets to include in the leaderboard.

    Returns:
        The DataFrame.
    """
    # Extract data
    data_dict: dict[str, list] = defaultdict(list)
    for model_id, results in model_results.items():
        total_results = {
            dataset: total_score for dataset, (_, total_score) in results.items()
        }
        metadata = metadata_dict[model_id]

        data_dict["model"].append(model_id)

        for category, rank in ranks[model_id].items():
            rank = round(rank, 2)
            data_dict[f"{category}_rank"].append(rank)

        default_dataset_values = {ds: float("nan") for ds in datasets} | {
            f"{ds}_version": "0.0.0" for ds in datasets
        }
        model_values = default_dataset_values | total_results | metadata
        for key, value in model_values.items():
            if isinstance(value, float):
                value = round(value, 2)
            data_dict[key].append(value)

        assert len({len(values) for values in data_dict.values()}) == 1, (
            f"Length of data_dict values must be equal, but got "
            f"{dict([(key, len(values)) for key, values in data_dict.items()])}."
        )

    # Sort categories, ensure that "nlu" is always first if present
    unique_categories = {
        category for model_ranks in ranks.values() for category in model_ranks.keys()
    }
    sorted_categories = list()
    if "nlu" in unique_categories:
        sorted_categories.append("nlu")
        unique_categories.remove("nlu")
    sorted_categories.extend(sorted(unique_categories))

    # Create dataframe and sort it
    df = (
        pd.DataFrame(data_dict)
        .sort_values(by=f"{sorted_categories[0]}_rank", ascending=True)
        .reset_index(drop=True)
    )

    # Replace infinite values with a large number, to allow sorting in web UI
    df = df.replace(to_replace=math.inf, value=999.00)

    # Reorder columns
    cols = [
        "model",
        *[f"{category}_rank" for category in sorted_categories],
        "parameters",
        "vocabulary_size",
        "context",
        "speed",
        "commercial",
        "merge",
    ]
    cols += [
        col for col in df.columns if col not in cols and not col.endswith("_version")
    ]
    cols += [col for col in df.columns if col not in cols and col.endswith("_version")]
    df = df[cols]

    assert isinstance(df, pd.DataFrame)
    return df


def significantly_better(
    score_values_1: list[float], score_values_2: list[float]
) -> float:
    """Compute one-tailed t-statistic for the difference between two sets of scores.

    Args:
        score_values_1:
            The first set of scores.
        score_values_2:
            The second set of scores.

    Returns:
        The t-statistic of the difference between the two sets of scores, where
        a positive t-statistic indicates that the first set of scores is
        statistically better than the second set of scores.
    """
    assert len(score_values_1) == len(score_values_2), (
        f"Length of score values must be equal, but got {len(score_values_1)} and "
        f"{len(score_values_2)}."
    )
    if score_values_1 == score_values_2:
        return 0
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        test_result = stats.ttest_ind(
            a=score_values_1,
            b=score_values_2,
            alternative="greater",
            equal_var=False,
        )
    return test_result.pvalue < 0.05  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
