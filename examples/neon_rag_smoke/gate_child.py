"""Clean-subprocess serialization gate — simulates the trainer container.

Loads a bundle's ``env-cls.pkl`` with ``cloudpickle.loads`` in a FRESH process
that has NEITHER the postgres-search path on ``sys.path`` NOR
``NEON_CORPUS_DSN_RO`` set, then instantiates the env and runs one lexical search
against the live corpus. A by-reference pickle of ``main``/``NeonRagEnv`` dies
here with ``ModuleNotFoundError: main``; a correct by-value pickle reconstructs
both classes and returns ranked hits off the BAKED dsn.

Invoked by run.py as:  python gate_child.py <pkl_path> <query>
Prints ``GATE_OK`` + ranked hits, or ``GATE_FAIL`` + reason; exits non-zero on
failure. Reads no creds — the RO dsn must ride the pickle.
"""

from __future__ import annotations

import os
import sys

import cloudpickle


def main() -> int:
    pkl_path, query = sys.argv[1], sys.argv[2]

    # Hard-prove the container conditions: env dsn absent, and NEITHER the env
    # module (rag_env) NOR postgres-search's `main` is importable — so a correct
    # load MUST come from a by-value pickle, not a by-reference re-import.
    os.environ.pop("NEON_CORPUS_DSN_RO", None)
    if "NEON_CORPUS_DSN_RO" in os.environ:
        print("GATE_FAIL: NEON_CORPUS_DSN_RO still set")
        return 1
    import importlib.util

    for mod in ("main", "rag_env"):
        if importlib.util.find_spec(mod) is not None:
            print(f"GATE_FAIL: {mod!r} is importable here — not a clean container test")
            return 1

    with open(pkl_path, "rb") as f:
        blob = f.read()

    try:
        env_class, constructor_args = cloudpickle.loads(blob)
    except ModuleNotFoundError as e:
        print(f"GATE_FAIL: by-reference pickle — {type(e).__name__}: {e}")
        return 1

    print(f"  unpickled env class : {env_class.__module__}.{env_class.__name__}")
    env = env_class(**constructor_args)
    hits = env._search.search(query, mode="lexical", top_k=3)
    if not hits:
        print("GATE_FAIL: reconstructed env returned zero hits (baked dsn not read)")
        return 1

    print(f"GATE_OK: {len(hits)} ranked hits via reconstructed env (baked dsn):")
    for i, h in enumerate(hits, 1):
        snippet = str(h.get("content", ""))[:70].replace("\n", " ")
        print(f"    {i}. src={h.get('source')}  score={h.get('score'):.3f}  | {snippet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
