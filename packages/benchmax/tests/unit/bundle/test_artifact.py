from __future__ import annotations

import dataclasses
import sys

import pytest

from benchmax.bundle import (
    Bundle,
    BundleMetadata,
    BundlingError,
    IncompatibleBenchmaxError,
    IncompatiblePythonError,
    bundle_digest,
    dump_bundle,
    load_bundle,
    validate_bundle_compatibility,
)
from benchmax.envs import BaseEnv


def _metadata(
    *,
    pip_dependencies: object = (),
    benchmax_version: str = "0.1.0",
) -> BundleMetadata:
    return BundleMetadata(
        pip_dependencies=pip_dependencies,  # type: ignore[arg-type]
        python_version="3.12",
        benchmax_version=benchmax_version,
        env_class_source="class ExampleEnv: ...\n",
    )


def test_metadata_canonicalizes_and_owns_dependency_collection() -> None:
    dependencies = [
        'Widget[B,a] < 2, >= 1 ; python_version < "3.13"',
        "requests >= 2",
    ]

    metadata = _metadata(pip_dependencies=dependencies)
    dependencies.append("late-mutation")

    assert metadata.pip_dependencies == (
        "requests>=2",
        'widget[a,b]<2,>=1; python_version < "3.13"',
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.pip_dependencies = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "dependencies",
    [
        "requests>=2",
        [""],
        [42],
        ["not a requirement ???"],
    ],
)
def test_metadata_rejects_invalid_dependency_declarations(dependencies: object) -> None:
    with pytest.raises((TypeError, ValueError), match="dependenc"):
        _metadata(pip_dependencies=dependencies)


@pytest.mark.parametrize(
    "dependencies",
    [
        ["foo==1", "foo==1"],
        ["foo==1", "Foo==2"],
        ['foo==1; python_version < "3.13"', 'foo==2; sys_platform == "linux"'],
        ["foo[a]>=1", "foo[b]<2"],
    ],
)
def test_metadata_rejects_repeated_distribution_targets(
    dependencies: list[str],
) -> None:
    with pytest.raises(ValueError, match="target 'foo' is declared more than once"):
        _metadata(pip_dependencies=dependencies)


def test_metadata_json_is_canonical() -> None:
    metadata = _metadata(pip_dependencies=["Requests >= 2"])

    assert metadata.to_json_bytes() == (
        b'{"benchmax_version":"0.1.0",'
        b'"env_class_source":'
        b'"class ExampleEnv: ...\\n",'
        b'"pip_dependencies":["requests>=2"],'
        b'"python_version":"3.12"}'
    )
    assert BundleMetadata.from_json_bytes(metadata.to_json_bytes()) == metadata


def test_metadata_rejects_unknown_keys() -> None:
    # Pre-0.2 bundles carried runtime digests and pickle checksums.
    legacy = (
        b'{"benchmax_runtime_digest":"legacy","pickled_sha256":"abc",'
        + _metadata().to_json_bytes()[1:]
    )
    with pytest.raises(ValueError, match="unsupported keys.*re-bundle"):
        BundleMetadata.from_json_bytes(legacy)


def test_bundle_digest_covers_pickle_and_canonical_metadata() -> None:
    baseline = Bundle(b"pickle", _metadata(pip_dependencies=["requests>=2"]))
    equivalent = Bundle(b"pickle", _metadata(pip_dependencies=["Requests >= 2"]))
    changed_pickle = Bundle(b"pickle-2", _metadata())
    changed_metadata = Bundle(
        b"pickle",
        _metadata(pip_dependencies=["requests>=2"], benchmax_version="0.2.0"),
    )

    assert bundle_digest(baseline) == bundle_digest(equivalent)
    assert bundle_digest(baseline) != bundle_digest(changed_pickle)
    assert bundle_digest(baseline) != bundle_digest(changed_metadata)
    assert len(bundle_digest(baseline)) == 64


def test_public_compatibility_check_requires_same_version_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    monkeypatch.setattr("benchmax.bundle._benchmax_version", lambda: "0.2.1")
    compatible = BundleMetadata(
        pip_dependencies=(),
        python_version=current_python,
        benchmax_version="0.2.0",
        env_class_source=None,
    )

    # Same major.minor: patch drift is allowed.
    validate_bundle_compatibility(compatible)
    validate_bundle_compatibility(
        dataclasses.replace(compatible, benchmax_version="0.2.9.dev3")
    )

    with pytest.raises(IncompatiblePythonError, match="Python 0.0"):
        validate_bundle_compatibility(
            dataclasses.replace(compatible, python_version="0.0")
        )
    with pytest.raises(IncompatibleBenchmaxError, match="major.minor"):
        validate_bundle_compatibility(
            dataclasses.replace(compatible, benchmax_version="0.1.2")
        )
    with pytest.raises(IncompatibleBenchmaxError, match="Cannot parse"):
        validate_bundle_compatibility(
            dataclasses.replace(compatible, benchmax_version="not-a-version")
        )
    monkeypatch.setattr("benchmax.bundle._benchmax_version", lambda: "unknown")
    with pytest.raises(IncompatibleBenchmaxError, match="Cannot verify"):
        validate_bundle_compatibility(compatible)


def test_load_checks_metadata_compatibility_before_unpickling() -> None:
    incompatible = Bundle(
        pickled=b"not a pickle",
        metadata=BundleMetadata(
            pip_dependencies=(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
            benchmax_version="0.0.0-incompatible",
            env_class_source=None,
        ),
    )

    with pytest.raises(IncompatibleBenchmaxError):
        load_bundle(incompatible)


def test_dump_bundle_rejects_invalid_dependencies_before_serializing() -> None:
    class MinimalEnv(BaseEnv):
        reward_keys = ("score",)

        async def create_dataset(self, split, base_dir):
            raise NotImplementedError

        async def compute_reward(self, rollout):
            return {"score": 0.0}

    with pytest.raises(BundlingError, match="Invalid pip_dependencies.*PEP 508"):
        dump_bundle(MinimalEnv, pip_dependencies=["not a requirement ???"])
