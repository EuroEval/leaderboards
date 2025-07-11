"""Generate leaderboard CSV files from the EuroEval results."""

import datetime as dt
import json
import logging
import math
import re
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from yaml import safe_load

from .link_generation import generate_task_link
from .result_loading import load_processed_results
from .result_processing import extract_model_metadata, group_results_by_model
from .score_computation import compute_ranks
from .utils import convert_to_float

logger = logging.getLogger(__name__)


def generate_leaderboard(
    leaderboard_config_path: Path, categories: list[Literal["all", "nlu"]], force: bool
) -> None:
    """Generate leaderboard CSV files from the EuroEval results.

    Args:
        leaderboard_config_path:
            The path to the leaderboard configuration file.
        categories:
            The categories of leaderboards to generate. Should be a list containing
            "all" and/or "nlu".
        force:
            Force the generation of the leaderboard, even if no updates are found.
    """
    leaderboard_title = leaderboard_config_path.stem.replace("_", " ").title()

    logger.info(f"Generating {leaderboard_title} leaderboard...")

    # Load configs
    with Path("task_config.yaml").open(mode="r") as f:
        task_config: dict[str, dict[str, str]] = safe_load(stream=f)
    with leaderboard_config_path.open(mode="r") as f:
        config: dict[str, list[str]] = safe_load(stream=f)

    # If the config consists of multiple languages, we extract a dictionary with config
    # for each constituent language
    configs: dict[str, dict[str, list[str]]] = dict()
    if "languages" in config:
        for language in config["languages"]:
            with Path(f"leaderboard_configs/{language}.yaml").open(mode="r") as f:
                configs[language] = safe_load(stream=f)
    else:
        configs = {leaderboard_config_path.stem: config}

    del config

    datasets = [
        dataset
        for config in configs.values()
        for task_datasets in config.values()
        for dataset in task_datasets
    ]

    # Load results and set them up for the leaderboard
    results = load_processed_results()
    results = [record for record in results if record["dataset"] in datasets]
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]] = (
        group_results_by_model(results=results, task_config=task_config)
    )
    ranks = compute_ranks(
        model_results=model_results, task_config=task_config, configs=configs
    )
    metadata_dict = extract_model_metadata(results=results)

    # Only include dataset columns in monolingual leaderboards
    include_dataset_columns = len(configs) == 1

    # Generate the leaderboard and store it to disk
    dfs = generate_dataframe(
        model_results=model_results,
        ranks=ranks,
        metadata_dict=metadata_dict,
        categories=categories,
        task_config=task_config,
        leaderboard_configs=configs,
        include_dataset_columns=include_dataset_columns,
    )

    for category, df in zip(categories, dfs):
        leaderboard_path = (
            Path("leaderboards") / f"{leaderboard_config_path.stem}_{category}.csv"
        )
        simplified_leaderboard_path = (
            Path("leaderboards")
            / f"{leaderboard_config_path.stem}_{category}_simplified.csv"
        )

        # Create the simplified leaderboard
        df_simplified = df.copy()
        df_simplified = df[
            [
                "model",
                "generative_type",
                "rank",
                "parameters",
                "vocabulary_size",
                "context",
                "commercial",
                "merge",
            ]
        ]
        df_simplified = df_simplified.map(
            lambda x: x.split("@@")[0] if isinstance(x, str) else x
        )
        df_simplified = df_simplified.query("rank != '-'")
        df_simplified = df_simplified.convert_dtypes()

        # Check if anything got updated
        new_records: list[str] = list()
        not_comparison_columns = ["rank"] + list(configs.keys())
        comparison_columns = [
            col for col in df.columns if col not in not_comparison_columns
        ]
        if leaderboard_path.exists():
            old_df = pd.read_csv(leaderboard_path, header=0, skiprows=1)
            old_df.columns = [
                re.sub(r"<a href=.*?>(.*?)</a>", r"\1", col) for col in old_df.columns
            ]
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
                        old_df[comparison_columns]
                        .query("model == @model_id")
                        .dropna()
                        .map(convert_to_float)
                    )
                    new_model_results = (
                        df[comparison_columns]
                        .query("model == @model_id")
                        .dropna()
                        .map(convert_to_float)
                    )
                    model_has_new_results = not np.all(
                        old_model_results.values == new_model_results.values
                    )
                    if model_has_new_results:
                        new_records.append(model_id)
        else:
            new_records = df.model.tolist()

        # Remove anchor tags from model names
        new_records = [
            re.sub(r"<a href=.*?>(.*?)</a>", r"\1", model) for model in new_records
        ]

        if new_records or force:
            top_header, second_header = create_leaderboard_headers(
                df=df, leaderboard_configs=configs
            )

            df.columns = top_header

            # Add second header as the first row
            df.loc[-1] = second_header
            df.index = df.index + 1
            df.sort_index(inplace=True)
            df = df.fillna("?")

            df.to_csv(leaderboard_path, index=False)
            df_simplified.to_csv(simplified_leaderboard_path, index=False)
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


def create_leaderboard_headers(
    df: pd.DataFrame | pd.Series, leaderboard_configs: dict[str, dict[str, list[str]]]
) -> tuple[list[str], list[str]]:
    """Create the leaderboard headers.

    The first header includes the task types (with links), and the second header
    contains the 'original' header but with html links to the datasets.

    Args:
        df:
            The dataframe.
        leaderboard_configs:
            The leaderboard configurations.

    Returns:
        The first and second header.
    """
    old_header = list(df.columns)

    all_datasets = []
    dataset_to_language = {}
    dataset_to_task_info = {}

    for language, tasks in leaderboard_configs.items():
        DATASET_LINK_TAG = (
            f"<a href='https://euroeval.com/datasets/{language}#"
            + "{anchor}'>{dataset}</a>"
        )

        language_datasets = list(chain.from_iterable(tasks.values()))
        all_datasets.extend(language_datasets)

        for dataset in language_datasets:
            dataset_to_language[dataset] = (language, DATASET_LINK_TAG)

        for task, datasets in tasks.items():
            for dataset in datasets:
                dataset_to_task_info[dataset] = (task, len(datasets))

    top_header = []
    second_header = []
    processed_tasks_per_language: dict[str, set[str]] = {}
    seen_version_col = False

    for id, col in enumerate(old_header):
        leaderboard_col = col.replace("_", "-")
        if leaderboard_col in all_datasets:
            language, DATASET_LINK_TAG = dataset_to_language[leaderboard_col]
            task, num_datasets = dataset_to_task_info[leaderboard_col]

            if language not in processed_tasks_per_language:
                processed_tasks_per_language[language] = set()

            if task in processed_tasks_per_language[language]:
                top_header.append("")
                second_header.append(
                    DATASET_LINK_TAG.format(anchor=leaderboard_col, dataset=col)
                )
                continue

            task_link = generate_task_link(id, task)
            if num_datasets > 1:
                task_link = f"~~~{task_link}~~~"

            top_header.append(task_link)
            second_header.append(
                DATASET_LINK_TAG.format(anchor=leaderboard_col, dataset=col)
            )
            processed_tasks_per_language[language].add(task)
        else:
            if "version" in col and not seen_version_col:
                top_header.append("<span style='visibility: hidden;'>hidden</span>")
                seen_version_col = True
            else:
                top_header.append("")

            second_header.append(col)

    # handle the first and second columns
    top_header[0] = (
        "<span style='font-size: 12px; font-weight: normal; opacity: 0.6;'>"
        "Task Type"
        "</span>"
    )
    top_header[1] = "<span style='visibility: hidden;'>dummy</span>"

    return top_header, second_header


def generate_dataframe(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    ranks: dict[str, dict[str, dict[str, float]]],
    metadata_dict: dict[str, dict],
    categories: list[Literal["all", "nlu"]],
    task_config: dict[str, dict[str, str]],
    leaderboard_configs: dict[str, dict[str, list[str]]],
    include_dataset_columns: bool,
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
        include_dataset_columns:
            Whether to include dataset columns in the DataFrame.

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
                        + f"@@{main_score:.2f}"
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
            .sort_values(
                by="rank",
                key=lambda series: series.map(
                    lambda x: float(x.split("@@")[1]) if isinstance(x, str) else x
                ),
            )
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
        cols = ["model", "generative_type"] + rank_cols
        cols += ["parameters", "vocabulary_size", "context", "commercial", "merge"]
        if include_dataset_columns:
            cols += [
                col
                for col in df.columns
                if col not in cols and not col.endswith("_version")
            ]
            cols += [
                col
                for col in df.columns
                if col not in cols and col.endswith("_version")
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
