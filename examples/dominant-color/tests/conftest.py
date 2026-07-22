import sys
from pathlib import Path

# main.py is a script, not an installed module; import it from this example
# only (installed example packages share one venv, so names must not collide).
sys.path.insert(0, str(Path(__file__).parents[1]))
