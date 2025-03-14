"""Generate leaderboard CSV files from the EuroEval results."""

import datetime as dt
import json
import logging
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import pandas as pd
import scipy.stats as stats
from yaml import safe_load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


warnings.simplefilter(action="ignore", category=RuntimeWarning)


@click.command()
@click.argument("leaderboard_config")
@click.option(
    "--force/--no-force",
    "-f",
    default=False,
    show_default=True,
    help="Force the generation of the leaderboard, even if no updates are found.",
)
@click.option(
    "--categories",
    "-t",
    multiple=True,
    default=["all", "nlu"],
    show_default=True,
    help="The categories of leaderboards to generate.",
)
def main(leaderboard_config: str | Path, force: bool, categories: tuple[str]) -> None:
    """Generate leaderboard CSV files from the EuroEval results.

    Args:
        leaderboard_config:
            The path to the leaderboard configuration file.
        force:
            Force the generation of the leaderboard, even if no updates are found.
        leaderboard_types:
            The leaderboard types to generate.
    """
    leaderboard_config = Path(leaderboard_config)
    leaderboard_title = leaderboard_config.stem.replace("_", " ").title()

    logger.info(f"Generating {leaderboard_title} leaderboard...")

    # Load configs
    with Path("task_config.yaml").open(mode="r") as f:
        task_config: dict[str, dict[str, str]] = safe_load(stream=f)
    with leaderboard_config.open(mode="r") as f:
        config: dict[str, list[str]] = safe_load(stream=f)

    # If the config consists of multiple languages, we extract a dictionary with config
    # for each constituent language
    configs: dict[str, dict[str, list[str]]] = dict()
    if "languages" in config:
        for language in config["languages"]:
            with Path(f"leaderboard_configs/{language}.yaml").open(mode="r") as f:
                configs[language] = safe_load(stream=f)
    else:
        configs = {leaderboard_config.stem: config}

    del config

    datasets = [
        dataset
        for config in configs.values()
        for task_datasets in config.values()
        for dataset in task_datasets
    ]

    # Load results and set them up for the leaderboard
    results = load_results(allowed_datasets=datasets)
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]] = (
        group_results_by_model(
            results=results, task_config=task_config, leaderboard_configs=configs
        )
    )
    ranks = compute_ranks(
        model_results=model_results, task_config=task_config, configs=configs
    )
    metadata_dict = extract_model_metadata(results=results)

    # Generate the leaderboard and store it to disk
    dfs = generate_dataframe(
        model_results=model_results,
        ranks=ranks,
        metadata_dict=metadata_dict,
        categories=categories,
        task_config=task_config,
        leaderboard_configs=configs,
    )

    for category, df in zip(categories, dfs):
        leaderboard_path = (
            Path("leaderboards") / f"{leaderboard_config.stem}_{category}.csv"
        )

        # Check if anything got updated
        new_records: list[str] = list()
        not_comparison_columns = ["rank"] + list(configs.keys())
        comparison_columns = [
            col for col in df.columns if col not in not_comparison_columns
        ]
        if leaderboard_path.exists():
            old_df = pd.read_csv(leaderboard_path)
            if any(col not in old_df.columns for col in comparison_columns):
                new_records = df.model.tolist()
            else:
                for model_id in set(df.model.tolist() + old_df.model.tolist()):
                    old_df_is_missing_columns = any(
                        col not in old_df.columns for col in comparison_columns
                    )
                    if old_df_is_missing_columns:
                        new_records.append(model_id)
                        continue

                    model_is_new = (
                        model_id in df.model.values
                        and model_id not in old_df.model.values
                    )
                    model_is_removed = (
                        model_id in old_df.model.values
                        and model_id not in df.model.values
                    )
                    if model_is_new or model_is_removed:
                        new_records.append(model_id)
                        continue

                    old_model_results = (
                        old_df[comparison_columns].query("model == @model_id").dropna()
                    )
                    new_model_results = (
                        df[comparison_columns].query("model == @model_id").dropna()
                    )
                    model_has_new_results = not np.all(
                        old_model_results.values == new_model_results.values
                    )
                    if model_has_new_results:
                        new_records.append(model_id)
        else:
            new_records = df.model.tolist()

        if new_records or force:
            df.to_csv(leaderboard_path, index=False)
            timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notes = dict(annotate=dict(notes=f"Last updated: {timestamp} CET"))
            with leaderboard_path.with_suffix(".json").open(mode="w") as f:
                json.dump(notes, f, indent=2)
                f.write("\n")
            if not new_records and force:
                logger.info(
                    f"Updated the {category!r} category of the {leaderboard_title} "
                    "leaderboard with no changes."
                )
            else:
                logger.info(
                    f"Updated the following {len(new_records):,} models in the "
                    f"{category!r} category of the {leaderboard_title} leaderboard: "
                    f"{', '.join(new_records)}"
                )
                pass
        else:
            logger.info(
                f"No updates to the {category!r} category of the {leaderboard_title} "
                "leaderboard."
            )


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
    leaderboard_configs: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, list[tuple[list[float], float, float]]]]:
    """Group results by model ID.

    Args:
        results:
            The processed results.
        task_config:
            The task configuration.
        leaderboard_configs:
            The leaderboard configurations.

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

            # Sometimes the raw scores are normalised to [0, 1], so we need to scale them
            # back to [0, 100]
            if max(raw_scores) <= 1:
                raw_scores = [score * 100 for score in raw_scores]

            model_scores[model_id][dataset].append((raw_scores, total_score, std_err))

    return model_scores


def compute_ranks(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    task_config: dict[str, dict[str, str]],
    configs: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute the ranks of the models.

    Args:
        model_results:
            The model results.
        task_config:
            The task configuration.
        configs:
            The leaderboard configurations for each language.

    Returns:
        The ranks of the models, per task category and per language.
    """
    all_datasets = {
        language: [
            dataset for task_datasets in config.values() for dataset in task_datasets
        ]
        for language, config in configs.items()
    }

    model_dataset_ranks: dict[str, dict[str, float]] = defaultdict(dict)
    for _, datasets in all_datasets.items():
        for dataset in datasets:
            dummy_scores: list[tuple[list[float], float, float]] = [
                ([], float("nan"), 0)
            ]
            model_dataset_scores = [
                (model_id, *scores.get(dataset, dummy_scores)[0])
                for model_id, scores in model_results.items()
            ]
            model_dataset_scores = sorted(
                [x for x in model_dataset_scores if not np.isnan(x[-2])],
                key=lambda x: x[-2],
                reverse=True,
            ) + [x for x in model_dataset_scores if np.isnan(x[-2])]
            stddev = np.std(
                [
                    score
                    for _, _, score, _ in model_dataset_scores
                    if not np.isnan(score)
                ]
            )

            rank_score = 1.0
            previous_scores: list[float] = list()
            for model_id, raw_scores, _, _ in model_dataset_scores:
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

    model_task_ranks: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for model_id, dataset_ranks in model_dataset_ranks.items():
        for language, config in configs.items():
            for task, datasets in config.items():
                model_task_ranks[model_id][language][task] = np.mean(
                    [
                        dataset_ranks[dataset]
                        for dataset in datasets
                        if dataset in dataset_ranks
                    ]
                ).item()

    categories = {
        task_config[task]["category"] for config in configs.values() for task in config
    } | {"all"}
    model_task_category_ranks: dict[str, dict[str, dict[str, float]]] = defaultdict(
        dict
    )
    for model_id, score_dict in model_task_ranks.items():
        for category in categories:
            language_scores = [
                np.mean(
                    [
                        score_dict[language][task]
                        for task in config
                        if task_config[task]["category"] == category
                        or category == "all"
                    ]
                ).item()
                for language, config in configs.items()
                if any(
                    task_config[task]["category"] == category or category == "all"
                    for task in config
                )
            ]
            model_rank_scores = dict(overall=np.mean(language_scores).item())
            if len(language_scores) > 1:
                model_rank_scores |= dict(zip(configs.keys(), language_scores))
            model_task_category_ranks[model_id][category] = model_rank_scores

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
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    ranks: dict[str, dict[str, dict[str, float]]],
    metadata_dict: dict[str, dict],
    categories: tuple[str],
    task_config: dict[str, dict[str, str]],
    leaderboard_configs: dict[str, dict[str, list[str]]],
) -> list[pd.DataFrame]:
    """Generate DataFrames from the model results.

    Args:
        model_results:
            The model results.
        ranks:
            The ranks of the models.
        metadata_dict:
            The metadata.
        categories:
            The categories of leaderboards to generate.
        task_config:
            The task configuration.
        leaderboard_configs:
            The leaderboard configurations.

    Returns:
        The DataFrames.
    """
    if model_results == {}:
        logger.error("No model results found, skipping leaderboard generation.")
        return list()

    # Mapping from category to dataset names
    category_to_datasets = {
        category: [
            dataset
            for config in leaderboard_configs.values()
            for task, task_datasets in config.items()
            for dataset in task_datasets
            if task_config[task]["category"] == category or category == "all"
        ]
        for category in categories
    }

    dfs: list[pd.DataFrame] = list()
    for category in categories:
        data_dict: dict[str, list] = defaultdict(list)
        for model_id, results in model_results.items():
            # Get the overall rank for the model
            rank = round(ranks[model_id][category]["overall"], 2)
            language_ranks = ranks[model_id][category]
            language_ranks.pop("overall")

            # Get the default values for the dataset columns
            default_dataset_values = {
                ds: float("nan") for ds in category_to_datasets[category]
            } | {f"{ds}_version": "-@@0" for ds in category_to_datasets[category]}

            # Get individual dataset scores for the model
            total_results = dict()
            for dataset in category_to_datasets[category]:
                if dataset in results:
                    scores = results[dataset]
                else:
                    scores = [(list(), float("nan"), 0)]
                main_score = scores[0][1]
                if not math.isnan(main_score):
                    score_str = (
                        " / ".join(
                            f"{total_score:,.2f} ± {std_err:,.2f}"
                            for _, total_score, std_err in scores
                        )
                        + f"@@{main_score}"
                    )
                else:
                    score_str = "-@@-1"
                total_results[dataset] = score_str

            # Filter metadata dict to only keep the dataset versions belonging to the
            # category
            metadata = {
                key: value
                for key, value in metadata_dict[model_id].items()
                if not key.endswith("_version")
                or key.replace("_version", "") in category_to_datasets[category]
            }

            # Add all the model values to the data dictionary
            model_values = (
                dict(model=model_id, rank=rank)
                | language_ranks
                | default_dataset_values
                | total_results
                | metadata
            )
            for key, value in model_values.items():
                if isinstance(value, float):
                    value = round(value, 2)
                data_dict[key].append(value)

            # Sanity check that all values have the same length
            assert len({len(values) for values in data_dict.values()}) == 1, (
                f"Length of data_dict values must be equal, but got "
                f"{dict([(key, len(values)) for key, values in data_dict.items()])}."
            )

        # Create dataframe and sort by rank
        df = (
            pd.DataFrame(data_dict)
            .sort_values(by="rank", ascending=True)
            .reset_index(drop=True)
        )

        # Ensure that inf values appear at the bottom
        rank_cols = ["rank"]
        if len(leaderboard_configs) > 1:
            rank_cols += list(leaderboard_configs.keys())

        # Convert rank to string, where {shown value}@@{sort value} to ensures that NaN
        # values appear at the bottom.
        for col in rank_cols:
            df[col] = [
                f"{value:.2f}@@{value:.2f}"
                if not math.isinf(value)
                else "-@@100"  # just a large number
                for value in df[col]
            ]

        # Replace dashes with underlines in all column names
        df.columns = df.columns.str.replace("-", "_")

        # Reorder columns
        cols = ["generative_type", "model"] + rank_cols
        cols += [
            "parameters",
            "vocabulary_size",
            "context",
            "commercial",
            "merge",
        ]
        cols += [
            col
            for col in df.columns
            if col not in cols and not col.endswith("_version")
        ]
        cols += [
            col for col in df.columns if col not in cols and col.endswith("_version")
        ]
        df = df[cols]

        # Replace Boolean values by ✓ and ✗
        boolean_columns = ["commercial", "merge"]
        for col in boolean_columns:
            df[col] = df[col].apply(lambda x: "✓" if x else "✗")

        # Replace generative_type with emojis
        generative_type_emoji_mapping = {
            "base": "🧠",
            "instruction_tuned": "📝",
            "reasoning": "🤔",
        }
        df["generative_type"] = df.generative_type.map(
            lambda x: generative_type_emoji_mapping.get(x, "🔍")
        )

        assert isinstance(df, pd.DataFrame)
        dfs.append(df)

    return dfs


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
