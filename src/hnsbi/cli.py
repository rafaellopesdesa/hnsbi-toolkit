"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ._version import __version__
from .config import ToolkitConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hnsbi",
        description="Hybrid neural simulation-based inference toolkit",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate-config",
        help="validate a YAML or JSON configuration",
    )
    validate.add_argument("config", type=Path)
    validate.add_argument("--print", action="store_true", dest="print_config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate-config":
        config = ToolkitConfig.load(arguments.config)
        if arguments.print_config:
            print(json.dumps(config.raw, indent=2, sort_keys=True))
        else:
            print(
                f"Valid hNSBI configuration: {arguments.config} "
                f"(schema {config.raw['schema_version']})"
            )
        return 0
    raise AssertionError("argparse accepted an unknown command.")


if __name__ == "__main__":
    raise SystemExit(main())
