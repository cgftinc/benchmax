import sys
from pathlib import Path

# main.py is primarily a script; import it from this example only.
sys.path.insert(0, str(Path(__file__).parents[1]))
# model_server is a test-local fake; importlib import mode never adds the
# tests dir to sys.path, so do it here.
sys.path.insert(0, str(Path(__file__).parent))
