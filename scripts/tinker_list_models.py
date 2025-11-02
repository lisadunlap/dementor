#!/usr/bin/env python3
"""
List available models from a Tinker service.

Requires TINKER_API_KEY to be present in the environment (e.g., via ~/.env).
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        print(
            "Error: TINKER_API_KEY not set in environment. Ensure ~/.env is sourced (see ~/.bashrc).",
            file=sys.stderr,
        )
        return 1

    import tinker  # type: ignore

    service_client = tinker.ServiceClient()
    capabilities = service_client.get_server_capabilities()

    print("Available models:")
    for item in capabilities.supported_models:
        print(f"- {item.model_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


