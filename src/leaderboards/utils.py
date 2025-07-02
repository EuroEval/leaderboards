"""Utility functions for the project."""

import re
import warnings

from scipy import stats


def convert_to_float(value: str | float) -> float | str:
    """Convert a value to float if possible.

    Args:
        value:
            The value to convert, can be a string or a float.

    Returns:
        The value converted to float if possible, otherwise returns the original value.
    """
    try:
        return float(value)
    except Exception:
        return value


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
            a=score_values_1, b=score_values_2, alternative="greater", equal_var=False
        )
    return test_result.pvalue < 0.05  # type: ignore[attr-defined]


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
        if not record.get("few_shot", True):
            model_notes.append("zero-shot")

    if record.get("validation_split", False):
        model_notes.append("val")

    if model_notes:
        model_id = f"{re.sub(r'</a>$', '', model_id)} ({', '.join(model_notes)})</a>"

    return model_id


def get_record_hash(record: dict) -> str:
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
