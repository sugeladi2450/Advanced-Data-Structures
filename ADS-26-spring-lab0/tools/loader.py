"""Load all tool modules from the tools directory.

Importing a tool module triggers its @registry.register decorator,
so tools become available in the registry.
"""

import importlib
from pathlib import Path


def load_all_tools(tools_dir: Path | None = None) -> None:
    """Import all *_tool.py modules under tools/, registering them with the registry."""
    if tools_dir is None:
        tools_dir = Path(__file__).parent

    for path in sorted(tools_dir.glob("*_tool.py")):
        importlib.import_module(f"tools.{path.stem}")
