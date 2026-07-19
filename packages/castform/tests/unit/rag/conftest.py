"""Add tests/ directory to sys.path so ``from fakes.chroma import ...`` works."""

import sys
from pathlib import Path

_tests_dir = str(Path(__file__).parent)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
