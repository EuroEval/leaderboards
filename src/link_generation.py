"""Generating links for models."""

import logging
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
from tqdm.auto import tqdm


tqdm.pandas(desc="Adding URLs to models")
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
    url = generate_hf_hub_url(model_id=model_id)
    if url is None:
        url = generate_openai_url(model_id=model_id)
    if url is None:
        url = generate_anthropic_url(model_id=model_id)
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
    if model_id in available_openai_models:
        return f"https://platform.openai.com/docs/models/{model_id}"
    else:
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
    else:
        return None
