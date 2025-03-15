"""Generating links for models."""

import logging
import re
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    LocalTokenNotFoundError,
    RepositoryNotFoundError,
)
import openai
from huggingface_hub import HfApi
from requests.exceptions import RequestException
from dotenv import load_dotenv
from anthropic import Anthropic


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_anchor_tag(model_id: str) -> str:
    """Generate an anchor tag for a model.

    Args:
        model_id:
            The model ID.

    Returns:
        The anchor tag for the model, or the model ID if the URL cannot be generated.
    """
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)

    model_id_without_revision = model_id.split("@")[0]

    url = generate_ollama_url(model_id=model_id_without_revision)
    if url is None:
        url = generate_hf_hub_url(model_id=model_id_without_revision)
    if url is None:
        url = generate_openai_url(model_id=model_id_without_revision)
    if url is None:
        url = generate_anthropic_url(model_id=model_id_without_revision)
    if url is None:
        logger.error(f"Could not find a URL for model {model_id_without_revision}.")

    return model_id if url is None else f"<a href='{url}'>{model_id}</a>"


def generate_hf_hub_url(model_id: str) -> str | None:
    """Generate a model URL for a model hosted on the Hugging Face Hub.

    Args:
        model_id:
            The Hugging Face model ID.

    Returns:
        The URL for the model on the Hugging Face Hub, or None if the model does not
        exist on the Hugging Face Hub.
    """
    hf_api = HfApi()
    try:
        hf_api.model_info(repo_id=model_id)
        return f"https://huggingface.co/{model_id}"
    except (
        GatedRepoError,
        LocalTokenNotFoundError,
        RepositoryNotFoundError,
        HFValidationError,
        RequestException,
        OSError,
    ):
        return None


def generate_openai_url(model_id: str) -> str | None:
    """Generate a model URL for a model hosted on OpenAI.

    Args:
        model_id:
            The OpenAI model ID.

    Returns:
        The URL for the model on OpenAI, or None if the model does not exist on OpenAI.
    """
    available_openai_models = [
        model_info.id for model_info in openai.models.list().data
    ]

    if model_id == "gpt-4-1106-preview":
        model_id_without_version_id = "gpt-4-turbo"
    else:
        model_id_without_version_id_parts: list[str] = []
        for part in model_id.split("-"):
            if re.match(r"^\d{2,}$", part):
                break
            model_id_without_version_id_parts.append(part)
        model_id_without_version_id = "-".join(model_id_without_version_id_parts)

    if (
        model_id in available_openai_models
        or model_id_without_version_id in available_openai_models
    ):
        return f"https://platform.openai.com/docs/models/{model_id_without_version_id}"
    return None


def generate_anthropic_url(model_id: str) -> str | None:
    """Generate a model URL for a model hosted on Anthropic.

    Args:
        model_id:
            The Anthropic model ID.

    Returns:
        The URL for the model on Anthropic, or None if the model does not exist on
        Anthropic.
    """
    client = Anthropic()
    available_anthropic_models = [
        model_info.id for model_info in client.models.list().data
    ]
    if model_id in available_anthropic_models:
        return "https://docs.anthropic.com/en/docs/about-claude/models/all-models"
    return None


def generate_ollama_url(model_id: str) -> str | None:
    """Generate a model URL for a model hosted on Ollama.

    Args:
        model_id:
            The Ollama model ID.

    Returns:
        The URL for the model on Ollama, or None if the model does not exist on Ollama.
    """
    if model_id.startswith("ollama/") or model_id.startswith("ollama_chat/"):
        model_id_without_prefix_and_suffix = re.sub(":.+$", "", model_id.split("/")[1])
        return f"https://ollama.com/library/{model_id_without_prefix_and_suffix}"
    return None
