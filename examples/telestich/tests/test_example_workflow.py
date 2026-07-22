from __future__ import annotations

import main as example
from benchmax.bundle import load_bundle
from benchmax.envs import Environment


def test_bundle_inputs_are_explicit_and_local_capture_is_automatic(monkeypatch):
    captured: dict = {}
    sentinel = object()

    def fake_dump_bundle(env_class, **kwargs):
        captured["env_class"] = env_class
        captured.update(kwargs)
        return sentinel

    import benchmax.bundle

    monkeypatch.setattr(benchmax.bundle, "dump_bundle", fake_dump_bundle)

    constructor_args = {"judge_base_url": "https://judge.example/v1"}
    assert example.build_training_bundle(constructor_args) is sentinel
    assert captured == {
        "env_class": example.TelestichEnv,
        "constructor_args": constructor_args,
        "pip_dependencies": example.RUNTIME_DEPENDENCIES,
    }


def test_gpu_launch_requires_explicit_confirmation(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    assert example.confirm_gpu_launch("telestich-test")

    monkeypatch.setattr("builtins.input", lambda _: "")
    assert not example.confirm_gpu_launch("telestich-test")


def test_real_bundle_roundtrip_uses_automatic_local_capture():
    constructor_args = {"judge_base_url": "https://judge.example/v1"}
    bundle = example.build_training_bundle(constructor_args)

    env_class, restored_args = load_bundle(bundle, instantiate=False)
    assert issubclass(env_class, Environment)
    assert env_class.__name__ == "TelestichEnv"
    assert restored_args == constructor_args
    assert bundle.metadata.pip_dependencies == (
        "english-words",
        "openai",
        "pronouncing",
        "wordfreq",
    )
