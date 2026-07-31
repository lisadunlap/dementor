from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tinker

from workflows import record_adapter_mapping

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a Tinker LoRA state by URI and save it under a friendly adapter name."
        )
    )
    parser.add_argument(
        "--state-uri",
        type=str,
        required=True,
        help="Tinker state URI, e.g. tinker://<uuid>/weights/final",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name available on the Tinker service.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Adapter name to save under (overwrites if already exists).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/tinker_adapters.json"),
        help="Path to local registry JSON to store name -> sampler model_path mapping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if "TINKER_API_KEY" not in os.environ:
        raise EnvironmentError(
            "Please set the TINKER_API_KEY environment variable before running."
        )

    service = tinker.ServiceClient()
    training = service.create_lora_training_client(base_model=args.base_model)
    training.load_state(args.state_uri).result()
    training.save_weights_and_get_sampling_client(name=args.name)
    print(f"Saved Tinker adapter as '{args.name}' from state '{args.state_uri}'.")

    sampler_path = args.state_uri
    if "/sampler_weights/" not in sampler_path:
        sampler_path = sampler_path.replace("/weights/", "/sampler_weights/")
    if not sampler_path.endswith("/final"):
        sampler_path = sampler_path.rstrip("/") + "/final"

    record_adapter_mapping(args.name, sampler_path, args.registry)


if __name__ == "__main__":
    main()
