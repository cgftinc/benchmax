import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Dict, List, Optional, Type

from benchmax.envs.base_env import BaseEnv
from benchmax.bundle.errors import ValidationError
from benchmax.bundle.payload import EnvPayload

logger = logging.getLogger(__name__)


def validate_payload(
    payload: EnvPayload,
    constructor_args: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Validate a packaged payload by running it in an isolated environment.

    Creates a temporary venv, installs the declared dependencies, loads the
    payload using the standard loader, and optionally runs a smoke test.

    Args:
        payload: The packaged EnvPayload to validate.
        constructor_args: If provided, instantiate the class and call list_tools().

    Returns:
        List of warning strings (non-fatal). Fatal issues raise ValidationError.
    """
    return _run_isolated_validation(payload, constructor_args)


def validate_structure(
    env_class: Type[BaseEnv],
    pip_dependencies: List[str],
) -> List[str]:
    """Run structural checks that don't require instantiation."""
    warnings_list: List[str] = []

    # Must be a class and a BaseEnv subclass
    if not isinstance(env_class, type) or not issubclass(env_class, BaseEnv):
        raise ValidationError(
            f"{env_class} is not a subclass of BaseEnv. "
            "Packaged classes must inherit from benchmax.envs.base_env.BaseEnv."
        )

    if env_class is BaseEnv:
        raise ValidationError(
            "Cannot bundle BaseEnv directly. Provide a concrete subclass."
        )

    # Check all abstract methods are implemented
    abstract_methods = getattr(env_class, "__abstractmethods__", frozenset())
    if abstract_methods:
        raise ValidationError(
            f"Class {env_class.__name__} has unimplemented abstract methods: "
            f"{', '.join(sorted(abstract_methods))}"
        )

    # Validate pip_dependencies format
    for dep in pip_dependencies:
        if not isinstance(dep, str) or not dep.strip():
            raise ValidationError(
                f"Invalid pip dependency: {dep!r}. Must be a non-empty string."
            )

    # Warn if stdlib modules are declared
    stdlib_names = getattr(sys, "stdlib_module_names", set())
    for dep in pip_dependencies:
        pkg_name = _extract_package_name(dep)
        if pkg_name in stdlib_names:
            warnings_list.append(
                f"'{dep}' appears to be a stdlib module and doesn't need "
                "to be in pip_dependencies."
            )

    # Warn if declared deps are not importable locally
    for dep in pip_dependencies:
        import_name = _extract_package_name(dep).replace("-", "_")
        try:
            importlib.import_module(import_name)
        except ImportError:
            warnings_list.append(
                f"Declared dependency '{dep}' (tried import as '{import_name}') "
                "is not importable locally. This may be fine if the import name "
                "differs from the package name."
            )

    # Python version check
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current != "3.12":
        warnings_list.append(
            f"benchmax requires Python 3.12 but you are running {current}. "
            "The remote machine may fail to unpickle."
        )

    return warnings_list


def _run_isolated_validation(
    payload: EnvPayload,
    constructor_args: Optional[Dict[str, Any]],
) -> List[str]:
    """Create a temp venv, install deps, unpickle the payload, and smoke test.

    Args:
        payload: The packaged EnvPayload containing the pickled class and dependencies.
        constructor_args: If provided, instantiate the class and call list_tools().

    Returns:
        List of warning strings (non-fatal). Fatal issues raise ValidationError.
    """
    warnings_list: List[str] = []
    venv_dir = None

    try:
        venv_dir = tempfile.mkdtemp(prefix="benchmax_validate_")
        venv_python = os.path.join(venv_dir, "bin", "python")

        # 1. Create venv (use --without-pip for environments like Colab that lack ensurepip)
        print("[validator] Creating isolated venv...")
        result = subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", venv_dir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            warnings_list.append(
                f"Failed to create venv for isolated validation: {result.stderr}"
            )
            return warnings_list

        # 2. Bootstrap pip using get-pip.py
        print("[validator] Downloading and installing pip...")
        get_pip_script = os.path.join(venv_dir, "get-pip.py")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '{get_pip_script}')",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            warnings_list.append(
                f"Failed to download get-pip.py: {result.stderr}"
            )
            return warnings_list

        result = subprocess.run(
            [venv_python, get_pip_script, "--quiet"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            warnings_list.append(
                f"Failed to install pip in isolated venv: {result.stderr}"
            )
            return warnings_list

        # 3. Install benchmax + declared deps from payload
        # benchmax is needed because the pickled class inherits from BaseEnv
        # (cloudpickle is a dependency of benchmax, so it's installed automatically)
        deps_to_install = ["benchmax"] + list(payload.pip_dependencies)
        print(f"[validator] Installing dependencies: {deps_to_install}")
        install_cmd = [venv_python, "-m", "pip", "install", "--quiet"] + deps_to_install
        result = subprocess.run(
            install_cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise ValidationError(
                f"Failed to install dependencies in isolated venv:\n{result.stderr}"
            )
        print("[validator] Dependencies installed successfully.")

        # 4. Write the pickled class bytes to temp file
        pickle_path = os.path.join(venv_dir, "env_class.pkl")
        with open(pickle_path, "wb") as f:
            f.write(payload.pickled_class)

        # 5. Write and run smoke test script using cloudpickle directly
        constructor_args_json = json.dumps(constructor_args) if constructor_args else "null"
        smoke_script = textwrap.dedent(f"""\
            import json
            import asyncio
            import cloudpickle

            with open({pickle_path!r}, "rb") as f:
                env_class = cloudpickle.load(f)

            constructor_args = json.loads({constructor_args_json!r})
            if constructor_args is not None:
                instance = env_class(**constructor_args)
                tools = asyncio.run(instance.list_tools())
                asyncio.run(instance.shutdown())
                print(f"OK: {{env_class.__name__}} instantiated, {{len(tools)}} tools")
            else:
                print(f"OK: {{env_class.__name__}} loaded (no constructor_args, skipped instantiation)")
        """)
        script_path = os.path.join(venv_dir, "smoke_test.py")
        with open(script_path, "w") as f:
            f.write(smoke_script)

        print("[validator] Running smoke test...")
        result = subprocess.run(
            [venv_python, script_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise ValidationError(
                f"Isolated smoke test failed:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}\n"
                "This usually means a dependency is missing from pip_dependencies."
            )

        print(f"[validator] {result.stdout.strip()}")

    except ValidationError:
        raise
    except subprocess.TimeoutExpired:
        warnings_list.append(
            "Isolated validation timed out. The venv setup or smoke test "
            "took too long."
        )
    except Exception as e:
        warnings_list.append(f"Isolated validation failed unexpectedly: {e}")
    finally:
        if venv_dir and os.path.exists(venv_dir):
            shutil.rmtree(venv_dir, ignore_errors=True)

    return warnings_list


def _extract_package_name(dep: str) -> str:
    """Extract the base package name from a pip dependency string."""
    for sep in ["[", ">=", "<=", "==", "~=", "!=", ">", "<"]:
        dep = dep.split(sep)[0]
    return dep.strip()
