import sys
from pathlib import Path

# main.py is a script, not an installed module; import it from this project.
sys.path.insert(0, str(Path(__file__).parents[1]))
