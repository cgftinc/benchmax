"""Build, gate, and upload the neon RAG gitlab-smoke training-env bundle.

Pipeline:
  1. build the bundle: ``dump_bundle(NeonRagEnv, constructor_args={search: NeonSearch
     (baked RO dsn)}, pip_dependencies=[...], local_modules=[rag_env, main])`` so
     BOTH the env module (rag_env) and postgres-search's ``main`` (SearchEnv) are
     pickled BY VALUE; ``NeonSearch`` stays by-reference (castform[neon] provides it
     in-container).
  2. serialization gate: write env-cls.pkl, then load it in a FRESH subprocess
     (neutral cwd, no postgres-search path, NEON_CORPUS_DSN_RO unset) and run one
     live lexical search. This simulates the trainer host; a by-reference pickle
     dies here.
  3. upload to STAGING (api.castform.dev — the CASTFORM_API_KEY is a staging key)
     via ``upload_training_run(run_name="neon-rag-gitlab-smoke")``.

Run inside the neon_rag_smoke uv env:
    uv run --project examples/neon_rag_smoke python examples/neon_rag_smoke/run.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Make rag_env (and, via its path hack, `main`) importable at build time.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import rag_env  # noqa: E402  (env module; pickled by value)
from rag_env import NeonRagEnv, neon_env_constructor_args  # noqa: E402

import main as postgres_search_main  # noqa: E402  (SearchEnv module; by value)

from benchmax.bundle import dump_bundle  # noqa: E402
from castform.platform.training_run import upload_training_run  # noqa: E402

CREDS = Path.home() / ".config" / "neon-benchmax.env"
RUN_NAME = "neon-rag-gitlab-smoke"
# Dataset files under datasets/; override to the validity-filtered large set
# (train_large.jsonl=400 / eval_large.jsonl=30) via env vars.
TRAIN_FILE = os.environ.get("NEON_TRAIN_FILE", "train.jsonl")
EVAL_FILE = os.environ.get("NEON_EVAL_FILE", "eval.jsonl")
# The CASTFORM_API_KEY is a STAGING key (api.castform.dev 200 / api.castform.com
# 401). The SDK default profile is prod, so target staging explicitly.
STAGING_BASE_URL = "https://api.castform.dev"
GATE_QUERY = "GitLab Security dashboard enablement link"

PIP_DEPENDENCIES = [
    "benchmax",
    "castform[neon]",
]


def load_creds() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in CREDS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_gate(pkl_bytes: bytes) -> None:
    """Load the pickle in a fresh, container-like subprocess and search live."""
    import os
    import shutil

    with tempfile.TemporaryDirectory() as tmp:
        pkl = Path(tmp) / "env-cls.pkl"
        pkl.write_bytes(pkl_bytes)
        # Copy the child INTO tmp and run it there, so Python's implicit
        # script-dir entry on sys.path is tmp (which holds neither rag_env nor
        # main) — a truly neutral container-like path. Drop NEON_CORPUS_DSN_RO
        # and PYTHONPATH from the child env.
        child = Path(tmp) / "gate_child.py"
        shutil.copy2(_HERE / "gate_child.py", child)
        child_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("NEON_CORPUS_DSN_RO", "PYTHONPATH")
        }
        proc = subprocess.run(
            [sys.executable, str(child), str(pkl), GATE_QUERY],
            cwd=tmp,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        print(proc.stdout, end="")
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(
                f"SERIALIZATION GATE FAILED (rc={proc.returncode}) — see above"
            )


def main() -> None:
    creds = load_creds()
    dsn = creds.get("NEON_CORPUS_DSN_RO")
    api_key = creds.get("CASTFORM_API_KEY")
    if not dsn:
        raise SystemExit("NEON_CORPUS_DSN_RO missing from creds")
    if not api_key:
        raise SystemExit("CASTFORM_API_KEY missing from creds")

    print("=" * 72)
    print("STEP 1: build bundle (NeonRagEnv + main pickled BY VALUE)")
    print("=" * 72)
    bundle = dump_bundle(
        NeonRagEnv,
        constructor_args=neon_env_constructor_args(dsn),
        pip_dependencies=PIP_DEPENDENCIES,
        local_modules=[rag_env, postgres_search_main],
    )
    print(f"  env class          : {NeonRagEnv.__module__}.{NeonRagEnv.__name__}")
    print(f"  pickled bytes      : {len(bundle.pickled)}")
    print(f"  pip_dependencies   : {list(bundle.metadata.pip_dependencies)}")

    print("\n" + "=" * 72)
    print("STEP 2: clean-subprocess serialization gate (simulates container)")
    print("=" * 72)
    run_gate(bundle.pickled)

    print("\n" + "=" * 72)
    print(f"STEP 3: upload to STAGING ({STAGING_BASE_URL})")
    print("=" * 72)
    train = load_jsonl(_HERE / "datasets" / TRAIN_FILE)
    eval_ = load_jsonl(_HERE / "datasets" / EVAL_FILE)
    print(f"  train rows / eval rows : {len(train)} / {len(eval_)}")
    uploaded = upload_training_run(
        bundle=bundle,
        train_dataset=train,
        eval_dataset=eval_,
        run_name=RUN_NAME,
        api_key=api_key,
        base_url=STAGING_BASE_URL,
    )

    print("\n" + "=" * 72)
    print("UPLOADED")
    print("=" * 72)
    print(f"  base_url           : {STAGING_BASE_URL}  (staging / app.castform.dev)")
    print(f"  env_cls_path       : {uploaded.env_cls_path}")
    print(f"  env_metadata_path  : {uploaded.env_metadata_path}")
    print(f"  dataset_path       : {uploaded.dataset_path}")
    print(f"  train.jsonl        : {uploaded.dataset_path}/train.jsonl")
    print(f"  eval.jsonl         : {uploaded.dataset_path}/eval.jsonl")
    print("  env-cls.args.pkl   : NOT produced (constructor_args ride env-cls.pkl)")


if __name__ == "__main__":
    main()
