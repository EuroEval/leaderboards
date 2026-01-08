"""Generate all leaderboards."""

import logging
import re
import warnings
from pathlib import Path
from typing import Literal

import click
from dotenv import load_dotenv

from leaderboards.leaderboard_generation import generate_leaderboard
from leaderboards.result_processing import process_results

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s ⋅ %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

warnings.simplefilter(action="ignore", category=RuntimeWarning)

load_dotenv()


# Constants for leaderboard generation
BANNED_VERSIONS: list[str] = ["9.3.0", "10.0.0"]
BANNED_MODEL_PATTERNS: list[re.Pattern] = [
    re.compile("^meta-llama/Llama-3.1-405B-Instruct$"),  # Temporary ban
    re.compile("^utter-project/EuroVLM-9B-Preview$"),  # Temporary ban
]
API_MODEL_PATTERNS: list[re.Pattern] = [
    re.compile(r"gemini/.*"),
    re.compile(r"(openai/)?gpt-[456789].*"),
    re.compile(r"(anthropic/)?claude.*"),
    re.compile(r"(xai/)?grok.*"),
]


@click.command()
@click.option(
    "--categories",
    "-c",
    default=["all", "nlu"],
    multiple=True,
    help="Categories to generate leaderboards for. Defaults to 'all' and 'nlu'.",
)
@click.option(
    "--force/--no-force",
    "-f",
    default=False,
    show_default=True,
    help="Force the generation of the leaderboard, even if no updates are found.",
)
def main(categories: tuple[Literal["all", "nlu"]], force: bool) -> None:
    """Generate all leaderboards.

    Args:
        categories:
            Categories to generate leaderboards for. Defaults to 'all' and 'nlu'.
        force:
            Whether to force the generation of the leaderboard, even if no updates are
            found.
    """
    process_results(
        banned_versions=BANNED_VERSIONS,
        banned_model_patterns=BANNED_MODEL_PATTERNS,
        api_model_patterns=API_MODEL_PATTERNS,
    )
    all_leaderboard_configs = Path("leaderboard_configs").glob("*.yaml")
    for config_path in all_leaderboard_configs:
        generate_leaderboard(
            leaderboard_config_path=config_path,
            categories=list(categories),
            force=force,
        )


if __name__ == "__main__":
    main()
