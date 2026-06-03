"""Local environment validation — catches training failures before job submission.

Simulates one mini training step locally (no GPU, no network) to verify
the env class contract matches what the trainer expects.
"""

from __future__ import annotations

import asyncio
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import cloudpickle

if TYPE_CHECKING:
    from .client import ValidationResult

_TOOL_TIMEOUT = 30.0

_shared_loop: asyncio.AbstractEventLoop | None = None


def _run_async(coro: Any, timeout: float = _TOOL_TIMEOUT) -> Any:
    """Run a coroutine with a timeout on the shared validation loop.

    All checks share one loop so that loop-bound resources created inside
    user code (e.g. httpx.AsyncClient for a judge LLM) stay attached to a
    live loop across checks. asyncio.run() would close the loop after each
    call, leaving those clients to schedule aclose() on a dead loop at GC
    time — surfacing as "Event loop is closed" warnings at shutdown.
    """
    global _shared_loop
    if _shared_loop is None or _shared_loop.is_closed():
        _shared_loop = asyncio.new_event_loop()
    return _shared_loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))


def _shutdown_shared_loop() -> None:
    """Close the shared loop, draining async generators and pending tasks."""
    global _shared_loop
    if _shared_loop is None or _shared_loop.is_closed():
        _shared_loop = None
        return
    try:
        _shared_loop.run_until_complete(_shared_loop.shutdown_asyncgens())
        pending = [t for t in asyncio.all_tasks(_shared_loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            _shared_loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
    finally:
        _shared_loop.close()
        _shared_loop = None


def _build_dummy_args(
    schema: dict[str, Any],
    query: str,
) -> dict[str, Any]:
    """Build dummy tool args from a JSON schema, respecting enums."""
    args: dict[str, Any] = {}
    for pname, pschema in schema.get("properties", {}).items():
        ptype = pschema.get("type", "string")
        if "enum" in pschema:
            args[pname] = pschema["enum"][0]
        elif pname in ("query", "text", "search_query"):
            args[pname] = query
        elif ptype == "string":
            args[pname] = "test"
        elif ptype == "integer":
            args[pname] = 10
        elif ptype == "number":
            args[pname] = 1.0
        elif ptype == "boolean":
            args[pname] = True
        elif ptype == "array":
            args[pname] = []
        elif ptype == "object":
            args[pname] = {}
    return args


def _ensure_nest_asyncio() -> None:
    """Patch asyncio for Jupyter notebooks (running event loop)."""
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        pass


def _run_local_checks(
    env_class: type,
    env_args: dict[str, Any],
    train_dataset: list[dict[str, Any]],
    eval_dataset: list[dict[str, Any]] | None = None,
    local_modules: list[ModuleType] | None = None,
) -> tuple[int, int]:
    """Run the local (no-network) contract checks for ``validate_env``.

    Mirrors how the trainer calls env methods — dataset_preprocess,
    load_dataset, list_tools/run_tool, compute_reward, a simulated rollout,
    and pickle round-trips. Prints colored progress as it goes.

    Returns:
        ``(passed, failed)`` check counts. The caller owns the shared event
        loop lifecycle (``_shutdown_shared_loop``) and the summary line.
    """
    _ensure_nest_asyncio()

    local_modules = local_modules or []
    for mod in local_modules:
        if not isinstance(mod, ModuleType):
            raise TypeError(
                f"local_modules must contain module objects, got "
                f"{type(mod).__name__}: {mod!r}"
            )

    if not train_dataset:
        print("  \u2717 train_dataset is empty")
        return (0, 1)

    examples = train_dataset[:5]
    passed = 0
    failed = 0

    print(
        "  \u26a0 Tools will be called with dummy args against"
        " real backends. Use a test backend if tools have"
        " side effects."
    )

    print("Environment Validation")

    # ── 1. dataset_preprocess ────────────────────────────────────
    preprocessed = None
    try:
        preprocessed = env_class.dataset_preprocess(examples[0])
        required = {"id", "prompt_messages"}
        if not isinstance(preprocessed, dict) or not required.issubset(preprocessed):
            missing = (
                required - set(preprocessed)
                if isinstance(preprocessed, dict)
                else required
            )
            print(
                f"  \u2717 dataset_preprocess did not return Example "
                f"(missing {sorted(missing)})"
            )
            print(
                "    Fix: return benchmax.envs.example_id.make_example("
                "prompt_messages=[...], task=...)."
            )
            failed += 1
        else:
            print(
                "  \u2713 dataset_preprocess returns Example with id + prompt_messages"
            )
            passed += 1
    except Exception as exc:
        print(f"  \u2717 dataset_preprocess raised {type(exc).__name__}: {exc}")
        failed += 1

    # ── 2. prompt_messages shape ───────────────────────────────────
    if (
        preprocessed
        and isinstance(preprocessed, dict)
        and "prompt_messages" in preprocessed
    ):
        seed = preprocessed["prompt_messages"]
        if not isinstance(seed, list) or not all(
            isinstance(mm, dict) and "role" in mm and "content" in mm for mm in seed
        ):
            print("  \u2717 prompt_messages is not a list of {role,content} dicts")
            failed += 1
        elif not seed:
            print("  \u2717 prompt_messages is empty")
            failed += 1
        else:
            print(f"  \u2713 prompt_messages is a {len(seed)}-message chat list")
            passed += 1
    else:
        print("  - prompt_messages shape check: skipped (no preprocessed result)")

    # ── 3. load_dataset ──────────────────────────────────────────
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
            tmp_path = f.name

        result = env_class.load_dataset("json", data_files=tmp_path, split="train")
        if isinstance(result, tuple) and len(result) == 2:
            ds, _ = result
            if len(ds) > 0:
                print(
                    f'  \u2713 load_dataset accepts ("json", data_files=...,'
                    f' split="train") — {len(ds)} rows'
                )
                passed += 1
            else:
                print("  \u2717 load_dataset returned empty dataset")
                failed += 1
        else:
            print(
                f"  \u2717 load_dataset returned {type(result).__name__},"
                " expected (Dataset, str | None)"
            )
            failed += 1

        Path(tmp_path).unlink(missing_ok=True)
    except Exception as exc:
        print(f"  \u2717 load_dataset raised {type(exc).__name__}: {exc}")
        print(
            '    Fix: load_dataset must accept ("json", data_files=path, split="train").'
        )
        failed += 1

    # ── 4. Instantiate env + list_tools + run_tool ───────────────
    env = None
    try:
        env = env_class(**env_args)
    except Exception as exc:
        print(f"  \u2717 env instantiation failed: {type(exc).__name__}: {exc}")
        failed += 1

    if env is not None:
        try:
            tools = _run_async(env.list_tools())
            print(f"  \u2713 list_tools returns {len(tools)} tool(s)")
            passed += 1

            if tools:
                tool = tools[0]
                dummy_args = _build_dummy_args(tool.input_schema, "test query")

                try:
                    result = _run_async(
                        env.run_tool(
                            rollout_id="test", tool_name=tool.name, **dummy_args
                        )
                    )
                    if isinstance(result, str):
                        print(f"  \u2713 run_tool returns string (tested: {tool.name})")
                        passed += 1
                    else:
                        print(
                            f"  \u2717 run_tool returned {type(result).__name__}, expected string"
                        )
                        failed += 1
                except Exception as exc:
                    print(f"  \u2717 run_tool raised {type(exc).__name__}: {exc}")
                    print(
                        "    Fix: run_tool must return a string. If tools need a real backend,"
                    )
                    print(
                        "    the training loop calls run_tool when the model generates tool_calls."
                    )
                    failed += 1
            else:
                print("  - run_tool: skipped (no tools defined)")
        except Exception as exc:
            print(f"  \u2717 list_tools raised {type(exc).__name__}: {exc}")
            failed += 1

    # ── 5. compute_reward with trainer-style args ────────────────
    if (
        env is not None
        and isinstance(preprocessed, dict)
        and "prompt_messages" in preprocessed
    ):
        try:
            task = preprocessed.get("task")
            init_args = preprocessed.get("init_rollout_args") or {}

            # Simulate the trainer's call: a fake one-turn transcript echoing
            # the seed plus a stub assistant turn. compute_reward receives
            # task verbatim and runtime kwargs (init_rollout_args fields).
            messages = list(preprocessed["prompt_messages"]) + [
                {
                    "role": "assistant",
                    "content": "I found the answer based on the search results.",
                }
            ]

            reward = _run_async(
                env.compute_reward(
                    rollout_id="test-rollout",
                    messages=messages,
                    task=task,
                    **init_args,
                )
            )

            if not isinstance(reward, dict):
                print(
                    f"  \u2717 compute_reward returned"
                    f" {type(reward).__name__}, expected dict[str, float]"
                )
                failed += 1
            else:
                bad_values = {
                    k: type(v).__name__
                    for k, v in reward.items()
                    if not isinstance(v, (int, float))
                }
                if bad_values:
                    print(f"  \u2717 compute_reward has non-float values: {bad_values}")
                    failed += 1
                else:
                    non_finite = {
                        k: v for k, v in reward.items() if not math.isfinite(v)
                    }
                    if non_finite:
                        print(
                            f"  \u2717 compute_reward has NaN/Inf values: {non_finite}"
                        )
                        print(
                            "    Fix: reward values must be finite."
                            " NaN/Inf break training gradients."
                        )
                        failed += 1
                    else:
                        print(
                            f"  \u2713 compute_reward returns dict[str, float]: {reward}"
                        )
                        passed += 1
        except Exception as exc:
            print(f"  \u2717 compute_reward raised {type(exc).__name__}: {exc}")
            print(
                "    Fix: compute_reward signature is (rollout_id, messages, task, **kwargs)."
            )
            print("    Read example data from `task`, runtime fields from `kwargs`.")
            failed += 1

    # ── 5b. Simulated rollout (E2E) ──────────────────────────────
    if (
        env is not None
        and isinstance(preprocessed, dict)
        and "prompt_messages" in preprocessed
    ):
        try:
            prompt_messages = preprocessed["prompt_messages"]
            task = preprocessed.get("task")
            init_args = preprocessed.get("init_rollout_args") or {}

            tools = _run_async(env.list_tools())

            # Seed text used for dummy tool-arg generation.
            first_user = next(
                (m["content"] for m in prompt_messages if m.get("role") == "user"),
                "",
            )
            query_text = first_user[:200] if isinstance(first_user, str) else "test"

            # ── Call each tool twice (catch stateful bugs) ──────
            transcript: list[dict[str, Any]] = list(prompt_messages)
            tool_call_count = 0
            for tool in tools:
                tool_args = _build_dummy_args(tool.input_schema, query_text)
                for _ in range(2):
                    result = _run_async(
                        env.run_tool(
                            rollout_id="sim-rollout",
                            tool_name=tool.name,
                            **tool_args,
                        )
                    )
                    transcript.append({"role": "assistant", "content": "Calling tool."})
                    transcript.append({"role": "tool", "content": str(result)[:500]})
                    tool_call_count += 1

            # Final assistant message echoing ground truth if available.
            gt = (task or {}).get("ground_truth")
            transcript.append(
                {"role": "assistant", "content": str(gt or "test answer")}
            )

            reward = _run_async(
                env.compute_reward(
                    rollout_id="sim-rollout",
                    messages=transcript,
                    task=task,
                    **init_args,
                )
            )

            if not isinstance(reward, dict):
                print(
                    f"  \u2717 simulated rollout: compute_reward returned {type(reward).__name__}"
                )
                failed += 1
            else:
                bad = {
                    k: v
                    for k, v in reward.items()
                    if not isinstance(v, (int, float)) or not math.isfinite(v)
                }
                if bad:
                    print(f"  \u2717 simulated rollout: bad reward values: {bad}")
                    failed += 1
                else:
                    tools_desc = (
                        f"{tool_call_count} tool calls"
                        if tool_call_count
                        else "no tools"
                    )
                    print(
                        f"  \u2713 simulated rollout OK ({tools_desc}, reward={reward})"
                    )
                    passed += 1

        except Exception as exc:
            print(f"  \u2717 simulated rollout failed: {type(exc).__name__}: {exc}")
            failed += 1

    # ── 6. Pickle round-trip ─────────────────────────────────────
    # Use cloudpickle on both sides so envs that import from local modules
    # (registered via local_modules in dump_bundle) round-trip the same way
    # as they will on the trainer. Plain pickle.loads can read simple cloudpickle
    # output but breaks on by-value module pickling — silently mismatching
    # local validation vs trainer behavior.
    # Register local_modules by value for the duration of the pickle checks,
    # mirroring what dump_bundle does. Without this, sibling-module envs get
    # pickled by reference and the local-modules guard below flags them \u2014
    # even though the user did everything right by listing them.
    try:
        for mod in local_modules:
            cloudpickle.register_pickle_by_value(mod)
        try:
            data = cloudpickle.dumps(env_class)
            restored_cls = cloudpickle.loads(data)
            restored_env = restored_cls(**env_args)
            tools = _run_async(restored_env.list_tools())
            print(
                f"  \u2713 pickle round-trip OK ({len(data)} bytes, {len(tools)} tools)"
            )
            passed += 1
        except Exception as exc:
            print(f"  \u2717 pickle round-trip failed: {type(exc).__name__}: {exc}")
            failed += 1

        # \u2500\u2500 6a. Local-modules guard \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        # Same-process round-trip above succeeds even when local_modules are
        # forgotten, because the user's working module is already in sys.modules
        # so cloudpickle's by-reference resolves via cache. On a fresh worker
        # process there's no cache and the import fails. Inspect the pickle's
        # find_class refs to catch this pre-upload.
        try:
            from benchmax.bundle import unregistered_local_refs

            risky = unregistered_local_refs(cloudpickle.dumps(env_class))
            if risky:
                print(
                    f"  \u2717 {env_class.__name__}: missing "
                    f"local_modules=[{', '.join(risky)}]"
                )
                print(
                    "    (round-trip above passed because sys.modules cache "
                    "hides this in-process; trainer will fail to import)"
                )
                failed += 1
            else:
                print("  \u2713 no unregistered local-module references")
                passed += 1
        except Exception as exc:
            print(f"  \u2717 local-modules check failed: {type(exc).__name__}: {exc}")
            failed += 1
    finally:
        for mod in local_modules:
            try:
                cloudpickle.unregister_pickle_by_value(mod)
            except Exception:
                pass

    # ── 6b. env_args pickle ────────────────────────────────────────
    try:
        args_data = cloudpickle.dumps(env_args)
        restored_args = cloudpickle.loads(args_data)
        assert isinstance(restored_args, dict)
        print(f"  \u2713 env_args pickle round-trip OK ({len(args_data)} bytes)")
        passed += 1
    except Exception as exc:
        print(f"  \u2717 env_args pickle failed: {type(exc).__name__}: {exc}")
        print(
            "    Fix: env_args must be serializable. Lambdas, SDK"
            " clients, and Pydantic models may not pickle."
        )
        failed += 1

    # ── 7. System prompt ─────────────────────────────────────────
    if env is not None:
        sp = getattr(env, "system_prompt", None)
        if not sp or not isinstance(sp, str):
            print("  warning: system_prompt is missing or not a string")
        else:
            msg = f"  \u2713 system_prompt: {len(sp)} chars"
            if len(sp) > 10000:
                msg += " (warning: very long — consider shortening)"
            print(msg)
        passed += 1

    return (passed, failed)


@dataclass(frozen=True)
class ValidationReport:
    """Combined outcome of :func:`validate_env` — local contract checks plus an
    optional remote smoke rollout.

    Bool-castable so the common gate keeps working::

        if not validate_env(env_class=Env, env_args={}, train_dataset=rows):
            raise SystemExit("Fix the env before launching.")

    Attributes:
        local_passed/local_failed: per-check counts from the local contract
            pass. Both 0 when ``local=False``.
        remote: the remote ``ValidationResult``, or ``None`` when no ``api_key``
            was given (remote smoke skipped).
        local_ran/remote_ran: which layers actually executed.
    """

    local_passed: int
    local_failed: int
    remote: ValidationResult | None
    local_ran: bool
    remote_ran: bool

    @property
    def local_ok(self) -> bool:
        return self.local_failed == 0

    @property
    def remote_ok(self) -> bool:
        return self.remote is None or self.remote.ok

    @property
    def ok(self) -> bool:
        # A report where nothing ran is NOT a pass — there was nothing to
        # validate, so refuse to green-light a launch (fail loudly).
        return (self.local_ran or self.remote_ran) and self.local_ok and self.remote_ok

    def __bool__(self) -> bool:
        return self.ok


def _print_report_summary(report: ValidationReport) -> None:
    """Print the one-line roll-up across whichever layers ran."""
    print()
    if not (report.local_ran or report.remote_ran):
        print(
            "  warning: validate_env ran no checks (local=False and no "
            "api_key) — nothing was validated."
        )
        return

    parts: list[str] = []
    if report.local_ran:
        total = report.local_passed + report.local_failed
        suffix = "" if report.local_ok else " — FAILED"
        parts.append(f"local {report.local_passed}/{total} checks{suffix}")
    if report.remote_ran and report.remote is not None:
        n = len(report.remote.examples)
        n_ok = sum(1 for ex in report.remote.examples if ex.ok)
        suffix = "" if report.remote_ok else " — FAILED"
        parts.append(f"remote {n_ok}/{n} rollouts{suffix}")

    status = "passed" if report.ok else "FAILED"
    print(f"validate_env {status}: " + "; ".join(parts))
    if not report.ok:
        print("  Fix the issues above before launching a training run.")


def validate_env(
    env_class: type,
    env_args: dict[str, Any],
    train_dataset: list[dict[str, Any]],
    eval_dataset: list[dict[str, Any]] | None = None,
    *,
    local_modules: list[ModuleType] | None = None,
    pip_dependencies: list[str] | None = None,
    local: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    remote_examples: int = 2,
    llm_model: str | None = None,
    max_turns: int = 4,
    verbose: bool = True,
) -> ValidationReport:
    """Validate an environment before launching a training run.

    A single entry point that folds together the two validation layers:

    1. **Local contract checks** (``local=True``, the default) — run in-process
       with no network, mirroring how the trainer calls env methods
       (dataset_preprocess, load_dataset, list_tools/run_tool, compute_reward,
       a simulated rollout, pickle round-trips).
    2. **Remote smoke rollout** (runs when ``api_key`` is given) — bundles the
       env inline and runs ``remote_examples`` real rollouts on the platform,
       exactly as :meth:`RolloutClient.validate_examples` does, but without you
       having to construct a client.

    Pass nothing but the env + dataset for a fast offline check; add
    ``api_key`` (and, if your script's environment differs from the SDK
    defaults, ``base_url``/``llm_base_url``) to also smoke-test against the
    platform::

        validate_env(env_class=Env, env_args={...}, train_dataset=rows)          # local only
        validate_env(env_class=Env, env_args={...}, train_dataset=rows,           # local + remote
                     api_key=API_KEY, base_url=BASE_URL, llm_base_url=LLM_URL)

    Generated launch scripts pass ``local=False`` because the trainer-image
    ``pip_dependencies`` may not be installed in the script's own environment;
    the remote rollout exercises the env on the trainer image instead.

    Warning:
        The local pass calls your env's tools with dummy arguments against real
        backends. If your tools have side effects (writes, deletes, sends), use
        a test backend.

    Args:
        env_class: The environment class (e.g. SearchEnv).
        env_args: Constructor kwargs for the env (bundled for the remote pass).
        train_dataset: Training examples (list of dicts). The remote pass
            smoke-tests the first ``remote_examples`` of these.
        eval_dataset: Optional eval examples (accepted for symmetry with the
            launch helpers; the local checks sample from ``train_dataset``).
        local_modules: Modules to pickle by value — pass the same list you give
            ``upload_training_run`` / ``dump_bundle`` for envs that import from
            sibling .py files.
        pip_dependencies: Pip deps recorded in the remote bundle.
        local: Run the local contract checks. Default True.
        api_key: Platform API key. Optional. The remote smoke runs whenever
            ``local=False`` (a launch) or an ``api_key`` is given; the credential
            then resolves from the key, or — when omitted — via the cached
            device-auth session (auto-login if interactive) / the ambient seam.
        base_url: Platform base URL for the remote pass. Defaults to
            ``config.platform_url()``.
        llm_base_url: Base URL for the rollout's LLM leg. Defaults to the
            platform LLM (``config.llm_url()``). Set it (together with
            ``llm_api_key``) to bring your own LLM — e.g.
            ``"https://api.openai.com/v1"``.
        llm_api_key: API key for the LLM leg. Required when ``llm_base_url``
            points outside the platform LLM endpoint (BYOK) — the platform key
            is never forwarded to a third-party host.
        remote_examples: Number of examples to smoke-test remotely (default 2).
        llm_model: Override the validation model for the remote rollout.
        max_turns: Max conversation turns per remote rollout.
        verbose: Print progress + the roll-up summary (default True).

    Returns:
        A bool-castable :class:`ValidationReport` carrying both layers' outcomes.
    """
    local_passed = local_failed = 0
    if local:
        try:
            local_passed, local_failed = _run_local_checks(
                env_class, env_args, train_dataset, eval_dataset, local_modules
            )
        finally:
            # _run_local_checks owns the shared validation loop; close it here
            # so loop-bound user resources (judge httpx clients) drain cleanly.
            _shutdown_shared_loop()

    remote: ValidationResult | None = None
    # Run the remote smoke when explicitly launching (local=False) or when an
    # api_key is passed. A keyless launch authenticates via the cached
    # device-auth session — auto-login here if interactive (no-op headless, where
    # RolloutClient then resolves via the ACT_AS_TOKEN_PATH / PLATFORM_API_KEY seam).
    run_remote = (not local) or bool(api_key)
    if run_remote:
        from benchmax.cli import ensure_session

        ensure_session()
        # Lazy import keeps the offline path free of httpx and avoids a
        # client <-> validation import cycle.
        from .client import RolloutClient

        rollout = RolloutClient(api_key=api_key, server_url=base_url)
        extra: dict[str, Any] = {}
        if llm_model is not None:
            extra["llm_model"] = llm_model
        remote = rollout.validate_examples(
            train_dataset,
            env_class=env_class,
            constructor_args=env_args,
            pip_dependencies=pip_dependencies,
            local_modules=local_modules,
            n=remote_examples,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key or "",
            max_turns=max_turns,
            verbose=verbose,
            **extra,
        )

    report = ValidationReport(
        local_passed=local_passed,
        local_failed=local_failed,
        remote=remote,
        local_ran=local,
        remote_ran=run_remote,
    )
    if verbose:
        _print_report_summary(report)
    return report
