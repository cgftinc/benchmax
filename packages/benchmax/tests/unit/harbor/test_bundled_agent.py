from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from harbor import DatasetConfig, EnvironmentType
from harbor.agents.factory import AgentFactory
from harbor.models.task.task import Task
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.tasks.client import TaskDownloadResult
from harbor.trial.trial import Trial

from benchmax.bundle import dump_bundle
from benchmax.envs.harbor import (
    BundledAgentSource,
    BundledHarborAgent,
    HarborEnv,
    HarborTrialTemplate,
)

_AGENT_SOURCE = b"""\
from pathlib import Path

from harbor.agents.base import BaseAgent
from .helper import MARKER

RESOURCE = Path(__file__).with_name("resource.txt").read_text(encoding="utf-8")

class CleanAgent(BaseAgent):
    resource = RESOURCE

    @staticmethod
    def name():
        return "clean-agent"

    def version(self):
        return MARKER

    async def setup(self, environment):
        pass

    async def run(self, instruction, environment, context):
        pass
"""


def _source_tree(root: Path) -> BundledAgentSource:
    root.mkdir()
    (root / "agent.py").write_bytes(_AGENT_SOURCE)
    (root / "helper.py").write_text('MARKER = "relative-import-worked"\n')
    (root / "resource.txt").write_text("adjacent-resource-worked")
    return BundledAgentSource.from_directory(
        root,
        files=("resource.txt", "helper.py", "agent.py"),
    )


def _agent(source: BundledAgentSource) -> BundledHarborAgent:
    return BundledHarborAgent(
        config=AgentConfig(
            import_path="agent:CleanAgent",
            kwargs={"marker_from_config": "preserved"},
        ),
        source=source,
    )


def test_source_identity_is_canonical_and_content_sensitive() -> None:
    first = BundledAgentSource.from_files(
        {"nested/helper.py": b"helper", "agent.py": b"agent"}
    )
    reordered = BundledAgentSource.from_files(
        {"agent.py": b"agent", "nested/helper.py": b"helper"}
    )
    changed = BundledAgentSource.from_files(
        {"agent.py": b"changed", "nested/helper.py": b"helper"}
    )

    assert first == reordered
    assert first.content_id == reordered.content_id
    assert first.content_id != changed.content_id
    assert first.files == (
        ("agent.py", b"agent"),
        ("nested/helper.py", b"helper"),
    )


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/agent.py",
        "../agent.py",
        "nested/../agent.py",
        "./agent.py",
        "a\\b.py",
        "agent\0.py",
    ],
)
def test_source_rejects_unsafe_or_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValueError, match="path"):
        BundledAgentSource.from_files({path: b"source"})


def test_source_rejects_duplicate_paths_and_symlinks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        BundledAgentSource((("agent.py", b"first"), ("agent.py", b"second")))

    outside = tmp_path / "outside.py"
    outside.write_text("pass")
    root = tmp_path / "source"
    root.mkdir()
    (root / "agent.py").symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        BundledAgentSource.from_directory(root, files=("agent.py",))

    real_package = tmp_path / "real-package"
    real_package.mkdir()
    (real_package / "nested.py").write_text("pass")
    (root / "package").symlink_to(real_package, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        BundledAgentSource.from_directory(root, files=("package/nested.py",))


def test_prepared_agent_config_is_small_and_json_serializable(tmp_path: Path) -> None:
    bundled = _agent(_source_tree(tmp_path / "author-source"))

    harbor_config = bundled._harbor_config()
    serialized_json = harbor_config.model_dump_json()
    serialized = json.loads(serialized_json)

    assert harbor_config.import_path.endswith(".agent:CleanAgent")
    assert bundled.source.content_id.removeprefix("sha256:") in (
        harbor_config.import_path
    )
    assert serialized["kwargs"] == {"marker_from_config": "preserved"}
    assert "adjacent-resource-worked" not in serialized_json
    assert bundled.config.import_path == "agent:CleanAgent"


def test_bundled_agent_load_is_concurrency_safe(tmp_path: Path) -> None:
    source = _source_tree(tmp_path / "author-source")
    config = BundledHarborAgent(
        config=AgentConfig(import_path="agent:CleanAgent"),
        source=source,
    )._harbor_config()

    def create(index: int):
        return AgentFactory.create_agent_from_config(
            config,
            logs_dir=tmp_path / f"logs-{index}",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        agents = list(executor.map(create, range(24)))

    modules = {type(agent).__module__ for agent in agents}
    assert modules == {
        f"_benchmax_harbor_agent_{source.content_id.removeprefix('sha256:')}.agent"
    }
    assert {agent.version() for agent in agents} == {"relative-import-worked"}
    assert {type(agent).__module__ for agent in agents} == modules
    assert {agent.resource for agent in agents} == {"adjacent-resource-worked"}


def test_source_revisions_coexist_without_modifying_sys_path(tmp_path: Path) -> None:
    first_source = BundledAgentSource.from_files(
        {
            "agent.py": _AGENT_SOURCE,
            "helper.py": b'MARKER = "first-revision"\n',
            "resource.txt": b"resource",
        }
    )
    second_source = BundledAgentSource.from_files(
        {
            "agent.py": _AGENT_SOURCE,
            "helper.py": b'MARKER = "second-revision"\n',
            "resource.txt": b"resource",
        }
    )
    original_sys_path = tuple(sys.path)

    first = AgentFactory.create_agent_from_config(
        BundledHarborAgent(
            config=AgentConfig(import_path="agent:CleanAgent"),
            source=first_source,
        )._harbor_config(),
        logs_dir=tmp_path / "first-logs",
    )
    second = AgentFactory.create_agent_from_config(
        BundledHarborAgent(
            config=AgentConfig(import_path="agent:CleanAgent"),
            source=second_source,
        )._harbor_config(),
        logs_dir=tmp_path / "second-logs",
    )

    assert first.version() == "first-revision"
    assert second.version() == "second-revision"
    assert type(first).__module__ != type(second).__module__
    assert tuple(sys.path) == original_sys_path


@pytest.mark.asyncio
async def test_harbor_trial_create_accepts_prepared_agent_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_tree(tmp_path / "author-source")
    agent_config = BundledHarborAgent(
        config=AgentConfig(import_path="agent:CleanAgent"),
        source=source,
    )._harbor_config()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text('schema_version = "1.3"\n')
    (task_dir / "instruction.md").write_text("Exercise the bundled agent")
    task = Task(task_dir, disable_verification=True)

    async def load_task(config: TrialConfig):
        return task, TaskDownloadResult(
            path=task_dir,
            download_time_sec=0,
            cached=True,
        )

    monkeypatch.setattr(Trial, "_load_task", staticmethod(load_task))
    monkeypatch.setattr(Trial, "_init_agent_environment", lambda self: None)
    monkeypatch.setattr(Trial, "_init_artifact_handler", lambda self: None)
    monkeypatch.setattr(Trial, "_validate_network_policy_modes", lambda self: None)

    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=task_dir),
            trial_name="bundled-agent-trial",
            trials_dir=tmp_path / "trials",
            agent=agent_config,
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=VerifierConfig(),
        )
    )

    assert trial.agent.name() == "clean-agent"
    assert trial.agent.version() == "relative-import-worked"
    assert trial.agent.resource == "adjacent-resource-worked"
    assert trial.agent.session_id == "bundled-agent-trial__agent"


def test_serialized_agent_loads_without_authoring_source_or_sys_path(
    tmp_path: Path,
) -> None:
    author_root = tmp_path / "author-source"
    bundled = _agent(_source_tree(author_root))
    bundle = dump_bundle(
        HarborEnv,
        constructor_args={
            "dataset": DatasetConfig(path=tmp_path / "dataset"),
            "reward_keys": ("reward",),
            "trial": HarborTrialTemplate(
                agent=bundled,
                environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
                verifier=VerifierConfig(),
            ),
        },
        pip_dependencies=("harbor>=0.18,<0.19",),
    )
    assert b"adjacent-resource-worked" in bundle.pickled

    pickle_path = tmp_path / "bundle.pkl"
    pickle_path.write_bytes(bundle.pickled)
    metadata_path = tmp_path / "bundle-metadata.json"
    metadata_path.write_bytes(bundle.metadata.to_json_bytes())
    shutil.rmtree(author_root)
    isolated = tmp_path / "isolated"
    isolated.mkdir()

    script = """
import sys
from pathlib import Path
from benchmax.bundle import Bundle, BundleMetadata, load_bundle
from harbor.agents.factory import AgentFactory

pickle_path, metadata_path, removed_source = map(Path, sys.argv[1:])
assert not removed_source.exists()
assert str(removed_source) not in sys.path
bundle = Bundle(
    pickled=pickle_path.read_bytes(),
    metadata=BundleMetadata.from_json_bytes(metadata_path.read_bytes()),
)
env = load_bundle(bundle)
bundled = env._trial.agent
agent = AgentFactory.create_agent_from_config(
    bundled._harbor_config(), logs_dir=Path("logs")
)
assert agent.version() == "relative-import-worked"
assert agent.resource == "adjacent-resource-worked"
assert type(agent).__module__.startswith("_benchmax_harbor_agent_")
print("clean-load-worked")
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(pickle_path),
            str(metadata_path),
            str(author_root),
        ],
        cwd=isolated,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "clean-load-worked"
