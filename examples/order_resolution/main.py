"""Command-line workflow for the order-resolution environment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from order_resolution.benchmark_spec import (
    BENCHMARK_ID,
    CANARY_AUTHORIZATION_PATH,
    HOSTED_VALIDATION_PATH,
    SPEC_PATH,
    assert_benchmark_id,
)
from order_resolution.preflight import PreflightError, run_preflight

DATA_DIR = Path(__file__).parent / "data"
V2_DATA_DIR = DATA_DIR / "v2"
EXAMPLE_ROOT = Path(__file__).parent
WORKTREE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_NEON_ENV = WORKTREE_ROOT / ".neon.env"
DEFAULT_NEON_MANIFEST = EXAMPLE_ROOT / "artifacts" / "neon.json"
DEFAULT_BASELINE_MANIFEST = EXAMPLE_ROOT / "artifacts" / "baseline.json"
DEFAULT_DEMO_ARTIFACT = EXAMPLE_ROOT / "artifacts" / "demo.json"
VALIDATION_MODEL = "gpt-5.4-mini"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    preflight = subparsers.add_parser(
        "preflight",
        help="verify the prepared worktree and record the audited implementation contract",
    )
    preflight.add_argument("--manifest", type=Path, required=True)

    test_local = subparsers.add_parser(
        "test-local",
        help="run migrations and selected tests against disposable local Postgres",
    )
    test_local.add_argument("--tests", nargs="+", default=["tests"])

    data = subparsers.add_parser("data", help="generate or check deterministic task data")
    data.add_argument("--force", action="store_true")
    data.add_argument("--check", action="store_true")
    data.add_argument("--seed", type=int, default=20260805)
    data.add_argument("--olist-calibration", type=Path)
    data.add_argument("--benchmark-id")

    contract_test = subparsers.add_parser(
        "contract-test",
        help="run model-free reward and sibling-isolation checks against local Postgres",
    )
    contract_test.add_argument("--benchmark-id")
    contract_test.add_argument("--compile-oracles", action="store_true")
    sync_parent = subparsers.add_parser(
        "sync-neon-parent",
        help="reconcile the content-addressed v2 catalog namespace in the approved parent",
    )
    sync_parent.add_argument("--benchmark-id", default=BENCHMARK_ID)
    sync_parent.add_argument("--spec", type=Path)
    sync_parent.add_argument("--neon-env", type=Path, default=DEFAULT_NEON_ENV)
    sync_parent.add_argument("--manifest", type=Path, default=DEFAULT_NEON_MANIFEST)
    setup_neon = subparsers.add_parser(
        "setup-neon",
        help="create, migrate, and seed the approved dedicated Neon project",
    )
    setup_neon.add_argument("--neon-env", type=Path, default=DEFAULT_NEON_ENV)
    setup_neon.add_argument("--manifest", type=Path, default=DEFAULT_NEON_MANIFEST)
    setup_neon.add_argument("--owner", default="Angel")

    validate = subparsers.add_parser(
        "validate",
        help="bundle and validate on an expiring least-privilege Neon child",
    )
    validate.add_argument("--hosted", action="store_true", required=True)
    validate.add_argument("--neon-env", type=Path, default=DEFAULT_NEON_ENV)
    validate.add_argument("--manifest", type=Path, default=DEFAULT_NEON_MANIFEST)
    validate.add_argument("--benchmark-id")
    validate.add_argument("--spec", type=Path, default=EXAMPLE_ROOT / SPEC_PATH)
    validate.add_argument("--output", type=Path)
    baseline = subparsers.add_parser(
        "baseline",
        help="run the frozen full baseline, stress, and signal-probe matrix",
    )
    baseline.add_argument("--manifest", type=Path, required=True)
    baseline.add_argument("--neon-env", type=Path, default=DEFAULT_NEON_ENV)
    baseline.add_argument("--neon-manifest", type=Path, default=DEFAULT_NEON_MANIFEST)
    probe_signal = subparsers.add_parser(
        "probe-signal",
        help="verify the signal probe already recorded by the frozen baseline run",
    )
    probe_signal.add_argument("--manifest", type=Path, required=True)
    report = subparsers.add_parser(
        "report",
        help="recompute JSON and HTML reports from frozen raw rollouts",
    )
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument(
        "--check",
        action="store_true",
        help="render to a temporary file and compare bytes without rewriting anything",
    )
    freeze = subparsers.add_parser(
        "freeze-benchmark",
        help="write the schema-v2 specification exactly once, before any model call",
    )
    freeze.add_argument("--benchmark-id", default=BENCHMARK_ID)
    freeze.add_argument("--predecessor", type=Path, default=DEFAULT_BASELINE_MANIFEST)
    freeze.add_argument("--spec", type=Path, default=EXAMPLE_ROOT / SPEC_PATH)
    benchmark = subparsers.add_parser(
        "benchmark",
        help="run one append-only v2 wave on a fresh disposable Neon child",
    )
    benchmark.add_argument("--wave", choices=("canary", "full"), required=True)
    benchmark.add_argument("--attempt", type=int)
    benchmark.add_argument("--spec", type=Path, default=EXAMPLE_ROOT / SPEC_PATH)
    benchmark.add_argument("--manifest", type=Path, required=True)
    benchmark.add_argument(
        "--authorization", type=Path, default=EXAMPLE_ROOT / CANARY_AUTHORIZATION_PATH
    )
    benchmark.add_argument("--requires-infrastructure-failure", type=Path)
    benchmark.add_argument("--requires-canary", type=Path)
    benchmark.add_argument("--neon-env", type=Path, default=DEFAULT_NEON_ENV)
    benchmark.add_argument("--neon-manifest", type=Path, default=DEFAULT_NEON_MANIFEST)
    verify_benchmark = subparsers.add_parser(
        "verify-benchmark",
        help="reconcile one sealed wave; --require-status/--require-decision are the gates",
    )
    verify_benchmark.add_argument("manifest", type=Path)
    verify_benchmark.add_argument("--require-status")
    verify_benchmark.add_argument("--require-decision")
    verify_report = subparsers.add_parser(
        "verify-report",
        help="reconcile every frozen rollout and aggregate report",
    )
    verify_report.add_argument("manifest", type=Path)
    verify_predecessor = subparsers.add_parser(
        "verify-predecessor",
        help="recheck the sealed v1 bytes and binding decision before any v2 work",
    )
    verify_predecessor.add_argument("--benchmark-id", default=BENCHMARK_ID)
    demo = subparsers.add_parser(
        "demo",
        help="replay the six frozen cases with redacted state and audit evidence",
    )
    demo.add_argument("--frozen-cases", action="store_true", required=True)
    demo.add_argument("--manifest", type=Path, default=DEFAULT_BASELINE_MANIFEST)
    demo.add_argument("--output", type=Path, default=DEFAULT_DEMO_ARTIFACT)
    branches = subparsers.add_parser(
        "branches",
        help="inspect disposable Neon branch cleanup",
    )
    branches.add_argument("--assert-clean", action="store_true", required=True)
    branches.add_argument("--neon-env", type=Path, default=DEFAULT_NEON_ENV)
    branches.add_argument("--manifest", type=Path, default=DEFAULT_NEON_MANIFEST)
    return parser


def _data(*, force: bool, check: bool, seed: int, olist_calibration: Path | None) -> int:
    from order_resolution.fixtures import check_data, read_olist_calibration, write_data

    if force and check:
        raise ValueError("data accepts only one of --force and --check")
    if olist_calibration is not None:
        summary = read_olist_calibration(olist_calibration)
        print(f"olist calibration: {json.dumps(summary, sort_keys=True)}")
    if check:
        hashes = check_data(DATA_DIR, seed=seed)
        print("data: ok (180 train / 90 eval; 20/10 per cell)")
    else:
        generated = write_data(DATA_DIR, seed=seed, force=force)
        hashes = generated.hashes
        print("data: wrote 180 train / 90 eval / 180 oracle traces")
    for name, digest in sorted(hashes.items()):
        print(f"{name}: sha256:{digest}")
    return 0


def _data_v2(*, force: bool, check: bool) -> int:
    from order_resolution.benchmark_spec import verify_predecessor
    from order_resolution.fixtures import check_v2_data, write_v2_data

    if force and check:
        raise ValueError("data accepts only one of --force and --check")
    verify_predecessor(EXAMPLE_ROOT, benchmark_id=BENCHMARK_ID)
    if check:
        hashes = check_v2_data(V2_DATA_DIR)
        print(f"data: ok ({BENCHMARK_ID}; 180 train / 90 eval; 20/10 per cell)")
    else:
        generated = write_v2_data(V2_DATA_DIR, force=force)
        hashes = generated.hashes
        print(f"data: wrote 180 train / 90 eval for {BENCHMARK_ID}")
    for name, digest in sorted(hashes.items()):
        print(f"{name}: sha256:{digest}")
    return 0


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=Path(__file__).parent, env=env, check=True)


def _test_local(tests: list[str]) -> int:
    example_root = Path(__file__).parent
    compose_file = example_root / "docker-compose.test.yml"
    project = f"benchmax-order-resolution-{os.getpid()}"
    compose = ["docker", "compose", "-p", project, "-f", str(compose_file)]
    try:
        _run([*compose, "up", "-d", "--wait"])
        port_result = subprocess.run(
            [*compose, "port", "postgres", "5432"],
            cwd=example_root,
            check=True,
            capture_output=True,
            text=True,
        )
        port = port_result.stdout.strip().rsplit(":", 1)[-1]
        if not port.isdigit():
            raise RuntimeError("could not resolve disposable Postgres port")
        database_url = (
            f"postgresql://order_resolution:order_resolution_test@127.0.0.1:{port}/order_resolution"
        )
        env = {
            **os.environ,
            "ORDER_RESOLUTION_ADMIN_DATABASE_URL": database_url,
            "ORDER_RESOLUTION_TEST_DATABASE_URL": database_url,
        }
        _run(["uv", "run", "alembic", "upgrade", "head"], env=env)
        _run(["uv", "run", "pytest", *tests, "-q"], env=env)
        return 0
    finally:
        subprocess.run(
            [*compose, "down", "-v", "--remove-orphans"],
            cwd=example_root,
            check=False,
        )


@contextmanager
def _disposable_postgres(purpose: str) -> Iterator[str]:
    """Bring up migrated throwaway Postgres and always tear the project down."""

    compose_file = EXAMPLE_ROOT / "docker-compose.test.yml"
    project = f"benchmax-order-resolution-{purpose}-{os.getpid()}"
    compose = ["docker", "compose", "-p", project, "-f", str(compose_file)]
    try:
        _run([*compose, "up", "-d", "--wait"])
        port_result = subprocess.run(
            [*compose, "port", "postgres", "5432"],
            cwd=EXAMPLE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        port = port_result.stdout.strip().rsplit(":", 1)[-1]
        if not port.isdigit():
            raise RuntimeError("could not resolve disposable Postgres port")
        database_url = (
            f"postgresql://order_resolution:order_resolution_test@127.0.0.1:{port}/order_resolution"
        )
        _run(
            ["uv", "run", "alembic", "upgrade", "head"],
            env={**os.environ, "ORDER_RESOLUTION_ADMIN_DATABASE_URL": database_url},
        )
        yield database_url
    finally:
        subprocess.run(
            [*compose, "down", "-v", "--remove-orphans"],
            cwd=EXAMPLE_ROOT,
            check=False,
        )


def _contract_test() -> int:
    from order_resolution.branching import run_seed_parent
    from order_resolution.contract import run_contract_test

    with _disposable_postgres("contract") as database_url:
        run_seed_parent(database_url)
        summary = asyncio.run(run_contract_test(database_url, DATA_DIR))
    print(f"contract-test: ok {json.dumps(summary, sort_keys=True)}")
    return 0


def _contract_test_v2(*, compile_oracles: bool) -> int:
    from order_resolution.benchmark_spec import verify_predecessor
    from order_resolution.branching import run_sync_parent_v2_catalog
    from order_resolution.contract import run_v2_contract_test

    verify_predecessor(EXAMPLE_ROOT, benchmark_id=BENCHMARK_ID)
    with _disposable_postgres("contract-v2") as database_url:
        run_sync_parent_v2_catalog(database_url)
        summary = asyncio.run(
            run_v2_contract_test(database_url, V2_DATA_DIR, compile_oracles=compile_oracles)
        )
    print(f"contract-test: ok {BENCHMARK_ID} {json.dumps(summary, sort_keys=True)}")
    return 0


def _sync_neon_parent(*, neon_env: Path, manifest_path: Path) -> int:
    from order_resolution.benchmark_spec import verify_predecessor
    from order_resolution.branching import (
        NeonApi,
        read_project_manifest,
        resolve_neon_api_key,
        run_sync_parent_v2_catalog,
    )
    from order_resolution.fixtures import build_v2_catalog

    verify_predecessor(EXAMPLE_ROOT, benchmark_id=BENCHMARK_ID)
    api_key = resolve_neon_api_key(neon_env)
    project = read_project_manifest(manifest_path)
    catalog = build_v2_catalog()
    with NeonApi(api_key) as api:
        admin_url = api.connection_uri(
            project,
            branch_id=project.parent_branch_id,
            role_name=project.admin_role_name,
            pooled=False,
        )
    outcome = run_sync_parent_v2_catalog(admin_url)
    print(
        f"sync-neon-parent: {outcome} namespace={catalog.id_prefix} "
        f"content=sha256:{catalog.content_sha256}"
    )
    return 0


def _setup_neon(*, neon_env: Path, manifest_path: Path, owner: str) -> int:
    from order_resolution.branching import (
        resolve_neon_api_key,
        run_seed_parent,
        setup_project,
    )

    stage = "credential loading"
    try:
        api_key = resolve_neon_api_key(neon_env)
        stage = "project creation"
        manifest, admin_url = setup_project(
            api_key=api_key,
            manifest_path=manifest_path,
            owner=owner,
        )
        stage = "migration"
        env = {**os.environ, "ORDER_RESOLUTION_ADMIN_DATABASE_URL": admin_url}
        _run(["uv", "run", "alembic", "upgrade", "head"], env=env)
        stage = "immutable catalog seed"
        run_seed_parent(admin_url)
    except Exception as error:
        raise RuntimeError(f"Neon setup failed during {stage}") from error
    print(f"setup-neon: ok project={manifest.project_id} parent={manifest.parent_branch_id}")
    print(f"manifest: {manifest_path.resolve()}")
    return 0


async def _run_validation(env, assets, *, base_dir: Path = DATA_DIR):
    from castform import validate_environment

    try:
        return await validate_environment(
            env,
            model=VALIDATION_MODEL,
            split="train",
            base_dir=base_dir,
            remote_assets=assets,
            max_context_tokens=8_192,
        )
    finally:
        await env.aclose()


def _validate_hosted(*, neon_env: Path, manifest_path: Path) -> int:
    from castform.platform import ensure_session, upload_assets
    from order_resolution.branching import (
        NeonApi,
        read_project_manifest,
        resolve_neon_api_key,
    )
    from order_resolution.fixtures import check_data
    from order_resolution.hosting import (
        build_environment_bundle,
        inspect_environment_bundle,
    )
    from order_resolution.order_env import OrderResolutionEnv

    api_key = resolve_neon_api_key(neon_env)
    manifest = read_project_manifest(manifest_path)
    check_data(DATA_DIR)
    record_path = EXAMPLE_ROOT / "artifacts" / "hosted-validation.json"
    branch = None
    record: dict[str, object] = {
        "project_id": manifest.project_id,
        "parent_branch_id": manifest.parent_branch_id,
        "model": VALIDATION_MODEL,
    }
    stage = "child branch creation"
    with NeonApi(api_key) as api:
        try:
            branch = api.create_runtime_branch(manifest, purpose="hosted-validation")
            record.update(
                {
                    "branch_id": branch.branch_id,
                    "branch_name": branch.branch_name,
                    "endpoint_id": branch.endpoint_id,
                    "expires_at": branch.expires_at,
                }
            )
            stage = "bundle inspection"
            bundle = build_environment_bundle(branch.runtime_database_url)
            inspection = inspect_environment_bundle(
                bundle,
                runtime_database_url=branch.runtime_database_url,
                forbidden_secrets={
                    "admin_url": branch.admin_database_url,
                    "api_key": api_key,
                },
            )
            record["bundle"] = inspection
            stage = "platform asset upload"
            ensure_session()
            assets = upload_assets(
                bundle=bundle,
                dataset_files={
                    "train.jsonl": DATA_DIR / "train.jsonl",
                    "eval.jsonl": DATA_DIR / "eval.jsonl",
                },
                run_name="order-resolution-neon-mvp-validation",
            )
            record["assets"] = {
                "env_cls_path": assets.env_cls_path,
                "env_metadata_path": assets.env_metadata_path,
                "dataset_path": assets.dataset_path,
            }
            stage = "local and remote rollouts"
            report = asyncio.run(
                _run_validation(OrderResolutionEnv(branch.runtime_database_url), assets)
            )
            record["ok"] = report.ok
            record["local_rollouts"] = len(report.local)
            record["remote_rollouts"] = len(report.remote or {})
            if not report.ok:
                raise RuntimeError("hosted validation did not satisfy the rollout contract")
        except Exception as error:
            raise RuntimeError(f"hosted validation failed during {stage}") from error
        finally:
            if branch is not None:
                api.delete_branch(manifest.project_id, branch.branch_id)
                record["deleted"] = True
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    print(
        "validate: ok "
        f"bundle={record['bundle']['digest']} "
        f"local={record['local_rollouts']} remote={record['remote_rollouts']}"
    )
    return 0


def _validate_hosted_v2(
    *, neon_env: Path, manifest_path: Path, spec_path: Path, output: Path
) -> int:
    from castform.platform import ensure_session, upload_assets
    from order_resolution.benchmark import (
        assert_bundle_matches_abi,
        assert_no_secrets,
        read_spec,
    )
    from order_resolution.benchmark_spec import verify_predecessor
    from order_resolution.branching import (
        NeonApi,
        read_project_manifest,
        resolve_neon_api_key,
    )
    from order_resolution.fixtures import check_v2_data
    from order_resolution.hosting import (
        build_environment_bundle,
        inspect_environment_bundle,
    )
    from order_resolution.order_env import OrderResolutionEnv

    verify_predecessor(EXAMPLE_ROOT, benchmark_id=BENCHMARK_ID)
    frozen, spec_sha256 = read_spec(spec_path)
    check_v2_data(V2_DATA_DIR)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing {output}")

    api_key = resolve_neon_api_key(neon_env)
    project = read_project_manifest(manifest_path)
    record: dict[str, object] = {
        "benchmark_id": BENCHMARK_ID,
        "spec_sha256": spec_sha256,
        "environment_abi_sha256": frozen["environment"]["abi_sha256"],
        "project_id": project.project_id,
        "parent_branch_id": project.parent_branch_id,
        "model": VALIDATION_MODEL,
    }
    branch = None
    stage = "child branch creation"
    with NeonApi(api_key) as api:
        try:
            branch = api.create_runtime_branch(project, purpose="v2-hosted-validation")
            record.update(
                {
                    "branch_id": branch.branch_id,
                    "branch_name": branch.branch_name,
                    "endpoint_id": branch.endpoint_id,
                    "expires_at": branch.expires_at,
                }
            )
            stage = "bundle and environment ABI verification"
            bundle = build_environment_bundle(branch.runtime_database_url)
            inspection = inspect_environment_bundle(
                bundle,
                runtime_database_url=branch.runtime_database_url,
                forbidden_secrets={
                    "admin_url": branch.admin_database_url,
                    "api_key": api_key,
                },
            )
            assert_bundle_matches_abi(
                inspection,
                example_root=EXAMPLE_ROOT,
                expected_abi_sha256=frozen["environment"]["abi_sha256"],
            )
            # Branch-specific by design; only this wave's manifest records it.
            record["bundle"] = inspection
            stage = "platform asset upload"
            ensure_session()
            assets = upload_assets(
                bundle=bundle,
                dataset_files={
                    "train.jsonl": V2_DATA_DIR / "train.jsonl",
                    "eval.jsonl": V2_DATA_DIR / "eval.jsonl",
                },
                run_name="order-resolution-v2-validation",
            )
            record["assets"] = {
                "env_cls_path": assets.env_cls_path,
                "env_metadata_path": assets.env_metadata_path,
                "dataset_path": assets.dataset_path,
            }
            stage = "local and remote rollouts"
            report = asyncio.run(
                _run_validation(
                    OrderResolutionEnv(branch.runtime_database_url), assets, base_dir=V2_DATA_DIR
                )
            )
            record["ok"] = report.ok
            record["local_rollouts"] = len(report.local)
            record["remote_rollouts"] = len(report.remote or {})
            if not report.ok:
                raise RuntimeError("hosted validation did not satisfy the rollout contract")
        except Exception as error:
            raise RuntimeError(f"hosted validation failed during {stage}") from error
        finally:
            if branch is not None:
                api.delete_branch(project.project_id, branch.branch_id)
                record["deleted"] = True
                record["deleted_at"] = _utc_now()
            serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
            for secret in (api_key, *( [branch.admin_database_url, branch.runtime_database_url]
                                       if branch is not None else [] )):
                if secret and secret in serialized:
                    raise RuntimeError("refusing to persist a secret-bearing validation record")
            assert_no_secrets(serialized, label=output.name)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
    print(
        "validate: ok "
        f"benchmark={BENCHMARK_ID} bundle={record['bundle']['digest']} "
        f"local={record['local_rollouts']} remote={record['remote_rollouts']} deleted=True"
    )
    print(f"record: {output.resolve()}")
    return 0


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _baseline(*, neon_env: Path, neon_manifest_path: Path, manifest_path: Path) -> int:
    from order_resolution.benchmark import run_baseline

    result = run_baseline(
        example_root=EXAMPLE_ROOT,
        neon_env=neon_env,
        neon_manifest_path=neon_manifest_path,
        output_manifest_path=manifest_path,
    )
    decision = result["report"]["decision"]
    print(f"baseline: complete rollouts={result['rollout_count']} decision={decision['status']}")
    print(f"manifest: {manifest_path.resolve()}")
    return 0


def _probe_signal(manifest_path: Path) -> int:
    from order_resolution.benchmark import verify_report_artifacts

    verified = verify_report_artifacts(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))["report"]["signal_probe"]
    print(
        "probe-signal: ok "
        f"groups={payload['groups']} mixed={payload['mixed_groups']} "
        f"passes={str(payload['passes']).lower()} decision={verified['decision']}"
    )
    return 0


def _report(manifest_path: Path, *, check: bool) -> int:
    from order_resolution.benchmark import check_v2_report, refresh_report_artifacts

    if check:
        result = check_v2_report(manifest_path, example_root=EXAMPLE_ROOT)
        print(f"report: unchanged {json.dumps(result, sort_keys=True)}")
        return 0
    manifest = refresh_report_artifacts(
        example_root=EXAMPLE_ROOT,
        manifest_path=manifest_path,
    )
    print(f"report: ok decision={manifest['report']['decision']['status']}")
    return 0


def _freeze_benchmark(*, predecessor: Path, spec_path: Path) -> int:
    from order_resolution.benchmark import freeze_v2_spec

    payload = freeze_v2_spec(
        example_root=EXAMPLE_ROOT, predecessor_manifest=predecessor, spec_path=spec_path
    )
    digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    print(
        f"freeze-benchmark: ok benchmark={payload['benchmark_id']} "
        f"spec=sha256:{digest} abi=sha256:{payload['environment']['abi_sha256']}"
    )
    print(f"spec: {spec_path.resolve()}")
    return 0


def _benchmark(args) -> int:
    from order_resolution.benchmark import run_v2_wave

    manifest = run_v2_wave(
        example_root=EXAMPLE_ROOT,
        spec_path=args.spec,
        manifest_path=args.manifest,
        wave=args.wave,
        attempt=args.attempt,
        authorization_path=args.authorization,
        requires_infrastructure_failure=args.requires_infrastructure_failure,
        requires_canary=args.requires_canary,
        neon_env=args.neon_env,
        neon_manifest_path=args.neon_manifest,
        run_nonce=secrets.token_hex(8),
    )
    decision = manifest.get("report", {}).get("decision", {}).get("status")
    print(
        f"benchmark: {args.wave} complete rollouts={manifest['rollout_count']} "
        f"status={manifest['status']}" + (f" decision={decision}" if decision else "")
    )
    print(f"manifest: {args.manifest.resolve()}")
    return 0


def _verify_benchmark(
    manifest_path: Path, *, require_status: str | None, require_decision: str | None
) -> int:
    from order_resolution.benchmark import verify_v2_benchmark

    result = verify_v2_benchmark(
        manifest_path,
        example_root=EXAMPLE_ROOT,
        require_status=require_status,
        require_decision=require_decision,
    )
    print(f"verify-benchmark: ok {json.dumps(result, sort_keys=True)}")
    return 0


def _verify_report(manifest_path: Path) -> int:
    from order_resolution.benchmark import verify_report_artifacts

    result = verify_report_artifacts(manifest_path)
    print(f"verify-report: ok {json.dumps(result, sort_keys=True)}")
    return 0


def _verify_predecessor(benchmark_id: str) -> int:
    from order_resolution.benchmark_spec import verify_predecessor

    result = verify_predecessor(EXAMPLE_ROOT, benchmark_id=benchmark_id)
    predecessor = result["predecessor"]
    print(
        "verify-predecessor: ok "
        f"benchmark={result['benchmark_id']} "
        f"predecessor={predecessor['benchmark_id']} "
        f"files={result['files_verified']} "
        f"rollouts={predecessor['rollout_count']} "
        f"decision={predecessor['decision']}"
    )
    return 0


def _demo(*, manifest_path: Path, output_path: Path) -> int:
    from order_resolution.branching import run_seed_parent, run_sync_parent_v2_catalog
    from order_resolution.demo import replay_frozen_demos

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    versioned = manifest.get("schema_version") == 2
    data_dir = V2_DATA_DIR if versioned else DATA_DIR

    compose_file = EXAMPLE_ROOT / "docker-compose.test.yml"
    project = f"benchmax-order-resolution-demo-{os.getpid()}"
    compose = ["docker", "compose", "-p", project, "-f", str(compose_file)]
    try:
        _run([*compose, "up", "-d", "--wait"])
        port_result = subprocess.run(
            [*compose, "port", "postgres", "5432"],
            cwd=EXAMPLE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        port = port_result.stdout.strip().rsplit(":", 1)[-1]
        if not port.isdigit():
            raise RuntimeError("could not resolve disposable Postgres port")
        database_url = (
            f"postgresql://order_resolution:order_resolution_test@127.0.0.1:{port}/order_resolution"
        )
        env = {**os.environ, "ORDER_RESOLUTION_ADMIN_DATABASE_URL": database_url}
        _run(["uv", "run", "alembic", "upgrade", "head"], env=env)
        run_seed_parent(database_url)
        if versioned:
            run_sync_parent_v2_catalog(database_url)
        artifact = asyncio.run(
            replay_frozen_demos(
                database_url=database_url,
                data_dir=data_dir,
                baseline_manifest_path=manifest_path,
                output_path=output_path,
            )
        )
    finally:
        subprocess.run(
            [*compose, "down", "-v", "--remove-orphans"],
            cwd=EXAMPLE_ROOT,
            check=False,
        )
    print(
        f"demo: ok cases={len(artifact['demos'])} decision={artifact['decision']} "
        f"artifact={output_path.resolve()}"
    )
    return 0


def _branches(*, neon_env: Path, manifest_path: Path) -> int:
    from order_resolution.branching import (
        NeonApi,
        read_project_manifest,
        resolve_neon_api_key,
    )

    api_key = resolve_neon_api_key(neon_env)
    manifest = read_project_manifest(manifest_path)
    with NeonApi(api_key) as api:
        branches = api.request("GET", f"/projects/{manifest.project_id}/branches").get(
            "branches", []
        )
    branch_ids = {str(branch.get("id")) for branch in branches}
    if branch_ids != {manifest.parent_branch_id}:
        unexpected = len(branch_ids - {manifest.parent_branch_id})
        raise RuntimeError(f"Neon branch cleanup failed: {unexpected} disposable branches remain")
    print(
        f"branches: clean project={manifest.project_id} parent={manifest.parent_branch_id} "
        f"delete_after={manifest.delete_after}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.action == "preflight":
            manifest = run_preflight(args.manifest)
            print(f"preflight: ok ({manifest['benchmax']['version']})")
            print(f"manifest: {args.manifest.resolve()}")
            return 0
        if args.action == "test-local":
            return _test_local(args.tests)
        if args.action == "data":
            if args.benchmark_id is not None:
                assert_benchmark_id(args.benchmark_id)
                return _data_v2(force=args.force, check=args.check)
            return _data(
                force=args.force,
                check=args.check,
                seed=args.seed,
                olist_calibration=args.olist_calibration,
            )
        if args.action == "contract-test":
            if args.benchmark_id is not None:
                assert_benchmark_id(args.benchmark_id)
                return _contract_test_v2(compile_oracles=args.compile_oracles)
            if args.compile_oracles:
                parser.error("--compile-oracles requires --benchmark-id order-resolution-v2")
            return _contract_test()
        if args.action == "sync-neon-parent":
            assert_benchmark_id(args.benchmark_id)
            return _sync_neon_parent(
                neon_env=args.neon_env,
                manifest_path=args.manifest,
            )
        if args.action == "setup-neon":
            return _setup_neon(
                neon_env=args.neon_env,
                manifest_path=args.manifest,
                owner=args.owner,
            )
        if args.action == "validate":
            if args.benchmark_id is not None:
                assert_benchmark_id(args.benchmark_id)
                return _validate_hosted_v2(
                    neon_env=args.neon_env,
                    manifest_path=args.manifest,
                    spec_path=args.spec,
                    output=args.output or (EXAMPLE_ROOT / HOSTED_VALIDATION_PATH),
                )
            return _validate_hosted(
                neon_env=args.neon_env,
                manifest_path=args.manifest,
            )
        if args.action == "baseline":
            return _baseline(
                neon_env=args.neon_env,
                neon_manifest_path=args.neon_manifest,
                manifest_path=args.manifest,
            )
        if args.action == "probe-signal":
            return _probe_signal(args.manifest)
        if args.action == "report":
            return _report(args.manifest, check=args.check)
        if args.action == "freeze-benchmark":
            assert_benchmark_id(args.benchmark_id)
            return _freeze_benchmark(predecessor=args.predecessor, spec_path=args.spec)
        if args.action == "benchmark":
            return _benchmark(args)
        if args.action == "verify-benchmark":
            return _verify_benchmark(
                args.manifest,
                require_status=args.require_status,
                require_decision=args.require_decision,
            )
        if args.action == "verify-report":
            return _verify_report(args.manifest)
        if args.action == "verify-predecessor":
            return _verify_predecessor(args.benchmark_id)
        if args.action == "demo":
            return _demo(
                manifest_path=args.manifest,
                output_path=args.output,
            )
        if args.action == "branches":
            return _branches(
                neon_env=args.neon_env,
                manifest_path=args.manifest,
            )
    except PreflightError as error:
        print(f"preflight: failed: {error}", file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    parser.error(f"unsupported action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
