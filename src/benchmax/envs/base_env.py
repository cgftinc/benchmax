import functools
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from benchmax.envs import capture
from benchmax.envs.example_id import canonical_example_id
from benchmax.envs.types import Example, Messages, ToolDefinition
from benchmax.prompts.tools import render_tools_prompt

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict


# Per-rollout env methods auto-wrapped by ``BaseEnv.__init_subclass__``.
# Signature: ``async def m(self, rollout_id, ...)``. Subclass overrides are
# wrapped in ``capture.maybe_capture_errors(rollout_id, name)`` so any
# exception lands in the per-rollout buffer when capture is installed.
_AUTOCAPTURE_SINGLE_RID: tuple[str, ...] = (
    "run_tool",
    "compute_reward",
    "init_rollout",
    "release_rollout",
    "copy_to_workspace",
    "copy_content_to_workspace",
    "copy_from_workspace",
)

# Group-reward gets a different wrap: the body spans multiple rids so there
# is no single ContextVar to bind, but a failure affects every rollout in
# the group, so the exception fans out one env_log per rid.
_AUTOCAPTURE_GROUP_RID: tuple[str, ...] = ("compute_group_reward",)

# Sentinel attribute marking an already-wrapped method so inheritance
# chains don't re-wrap (which would nest capture contexts multiple deep —
# safe, but pointless).
_AUTOCAPTURE_SENTINEL = "__benchmax_auto_captured__"


def _wrap_single(method: Callable, name: str) -> Callable:
    @functools.wraps(method)
    async def wrapper(self, rollout_id, *args, **kwargs):
        async with capture.maybe_capture_errors(rollout_id, name):
            return await method(self, rollout_id, *args, **kwargs)
    setattr(wrapper, _AUTOCAPTURE_SENTINEL, True)
    return wrapper


def _wrap_group(method: Callable, name: str) -> Callable:
    @functools.wraps(method)
    async def wrapper(self, rollout_ids, *args, **kwargs):
        async with capture.maybe_capture_group_errors(rollout_ids, name):
            return await method(self, rollout_ids, *args, **kwargs)
    setattr(wrapper, _AUTOCAPTURE_SENTINEL, True)
    return wrapper


def _apply_autocapture(cls: type) -> None:
    """Wrap any per-rollout env method defined directly on ``cls`` with the
    capture helper. Re-wrap is a no-op (sentinel attr short-circuits)."""
    for name in _AUTOCAPTURE_SINGLE_RID:
        method = cls.__dict__.get(name)
        if method is None or getattr(method, _AUTOCAPTURE_SENTINEL, False):
            continue
        setattr(cls, name, _wrap_single(method, name))
    for name in _AUTOCAPTURE_GROUP_RID:
        method = cls.__dict__.get(name)
        if method is None or getattr(method, _AUTOCAPTURE_SENTINEL, False):
            continue
        setattr(cls, name, _wrap_group(method, name))


class BaseEnv(ABC):
    """Base benchmax environment for tool execution and reward computation.

    Auth contract (G2):
        Subclasses that issue HTTP to platform services MUST call
        ``self.auth_headers(url)`` rather than reading ``os.environ`` directly.
        The default ``auth_headers`` is a no-op (safe for standalone/test use).
        Training infrastructure injects a live implementation via
        ``trainer.data.async_workers.orchestrator._wrap_with_castform_auth``, which
        monkeypatches the instance before the env is handed to a Ray actor.

    Logging contract:
        Use standard ``logging.getLogger(__name__)`` from any env method.
        Records emitted while the trainer is running will be captured
        per-rollout and shipped as ``rollout_env_logs`` rows. ``logger.exception``
        carries the traceback through. No need to import anything benchmax-
        specific or pass ``rollout_id`` to logging calls.
    """

    system_prompt: str = ""

    # Env-declared hints for training infra. None = use system defaults.
    recommended_max_turns: Optional[int] = None
    recommended_max_tool_calls: Optional[int] = None

    def __init__(self, **kwargs):
        pass

    def __init_subclass__(cls, **kwargs):
        """Wrap per-rollout method overrides with capture auto-attribution +
        warn when subclasses override ``auth_headers`` (easy to misunderstand).

        Auto-wrap: any override of a method in :data:`_AUTOCAPTURE_SINGLE_RID`
        / :data:`_AUTOCAPTURE_GROUP_RID` gets wrapped in
        :func:`capture.maybe_capture_errors` / :func:`capture.maybe_capture_group_errors`
        at class-definition time. Subclass authors keep writing
        ``async def run_tool(...)`` as before — the wrapper is added
        transparently, and is a no-op when capture is not installed.
        Sentinel attr on the wrapped callable prevents double-wrap on
        inheritance chains.

        Auth: the trainer monkeypatches ``auth_headers`` at the *instance*
        level, which always wins over class-level overrides. A subclass that
        defines its own ``auth_headers`` is dead code on the trainer.
        """
        super().__init_subclass__(**kwargs)
        _apply_autocapture(cls)
        if "auth_headers" in cls.__dict__ and cls.__dict__["auth_headers"] is not BaseEnv.auth_headers:
            logger.warning(
                "%s overrides auth_headers; the trainer will replace this at "
                "the instance level, so the override has no effect on training "
                "rollouts. Issue HTTP via self.auth_headers(url) instead of "
                "reading os.environ directly.",
                cls.__name__,
            )

    def auth_headers(self, url: str) -> dict[str, str]:
        """No-op by default. Override in a host-app mixin/subclass to attach
        bearer tokens to outbound platform-service calls. Custom envs MUST call
        self.auth_headers(url) when issuing HTTP to internal services rather
        than reading os.environ directly.
        """
        return {}


    # Override this method if your dataset rows aren't already shaped as
    # ``{seed_messages, task, init_rollout_args}``.
    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs) -> Example:
        """Preprocess a single dataset row into an :class:`Example`.

        The default implementation assumes ``example`` already has
        ``seed_messages`` (and optionally ``task`` / ``init_rollout_args``) and
        just computes the canonical ``id``. Envs whose source datasets have a
        different shape (HuggingFace columns, jsonl with custom keys, etc.)
        should override this to build the ``seed_messages`` chat list and
        ``task`` dict.

        Treats ``example`` as read-only — the original is never mutated.
        """
        if "seed_messages" not in example:
            raise ValueError(
                f"{cls.__name__}.dataset_preprocess: row is missing "
                "'seed_messages'. Override dataset_preprocess to build it "
                "from your dataset's columns."
            )
        seed_messages = example["seed_messages"]
        task = example.get("task")
        return Example(
            id=canonical_example_id(seed_messages, task),
            seed_messages=seed_messages,
            task=task,
            init_rollout_args=example.get("init_rollout_args"),
        )

    @classmethod
    def load_dataset(
        cls, dataset_name: str, **kwargs
    ) -> Tuple[
        "DatasetDict | Dataset | IterableDatasetDict | IterableDataset", str | None
    ]:
        """
        Download and prepare a dataset for use with this environment.

        This method should handle retrieving the specified dataset (e.g., from HuggingFace, local files,
        or a custom source), preprocessing or converting it into a compatible structure, and storing it
        locally in a reusable format. The processed dataset should be suitable for downstream use with
        `dataset_preprocess`, which standardizes individual examples into the expected format.

        Args:
            dataset_name (str): Identifier of the dataset to be loaded.
            **kwargs: Additional dataset-specific arguments (e.g., split, filtering options, cache directory).

        Returns:
            Dataset: A dataset object (e.g., HuggingFace Dataset or similar) ready for processing.
            str: Optional string pointing to where the dataset is stored locally
        """
        from datasets import load_dataset

        return load_dataset(dataset_name, **kwargs), None

    # Methods all environment subclasses must implement.
    #
    # Per-rollout methods (run_tool / compute_reward / init_rollout /
    # release_rollout / copy_* / compute_group_reward) are auto-wrapped by
    # ``__init_subclass__`` with ``capture.maybe_capture_errors`` so any
    # exception from a subclass override is attributed to the rollout
    # automatically when ``capture.install_capture()`` has been called.
    # Subclass authors don't need to opt in — just write the method
    # normally and standard ``logging`` from the body lands in the
    # rollout's env_log.

    @abstractmethod
    async def list_tools(self) -> List[ToolDefinition]:
        """Return list of available tools"""
        pass

    @abstractmethod
    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        """Execute named tool in rollout context with given arguments"""
        pass

    @abstractmethod
    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute rewards for a single rollout.

        Args:
            rollout_id: Identifier for the rollout being graded.
            messages: Full conversation transcript (seed messages + every assistant /
                tool turn produced during the rollout). Was named ``completion`` in
                the legacy signature, which was misleading: this is the entire
                message list, not just the model's final answer.
            task: Per-example reward-side data from
                :class:`benchmax.envs.types.Example` (e.g. ``ground_truth``,
                scoring config). ``None`` for envs that grade without per-row data.
            **kwargs: Trainer-runtime context (e.g. ``workspace_path``). Example
                fields should be read from ``task``, not ``kwargs``.

        Returns:
            Mapping of reward function name to scalar score.
        """
        pass

    async def compute_group_reward(
        self,
        rollout_ids: List[str],
        messages_list: List[Messages],
        tasks: List[Optional[Dict[str, Any]]],
        **kwargs: Any,
    ) -> List[Dict[str, float]]:
        """Compute rewards across a group of rollouts jointly.

        Override this when reward computation requires cross-rollout context (e.g.,
        relative scoring, group normalization, deduplication). Default returns
        empty dicts so per-rollout ``compute_reward`` runs in isolation.

        Args:
            rollout_ids: Identifiers for each rollout in the group.
            messages_list: One message transcript per rollout (see
                :meth:`compute_reward` for the renaming from ``completions``).
            tasks: One task dict (or ``None``) per rollout, paired by index.
            **kwargs: Trainer-runtime context shared across the group.

        Returns:
            One reward dict per rollout, in input order. An empty dict signals
            that no group reward was computed for that rollout.

        Note: failures fan out one ``logger.exception`` per rid (see
        :data:`_AUTOCAPTURE_GROUP_RID`), since a group-level error affects
        every rollout in the group. Internal logging from the body is NOT
        auto-captured per-rollout (multiple rids in flight, no single
        ContextVar binding) — surface per-rollout reasoning from
        :meth:`compute_reward` if you need it in ``rollout_env_logs``.
        """
        return [{} for _ in rollout_ids]

    async def get_system_prompt(self, add_tool_defs: bool = False) -> str:
        """Get system prompt. To add tool definitions, set add_tool_defs to True."""
        if add_tool_defs:
            return render_tools_prompt(
                await self.list_tools(), self.system_prompt or ""
            )
        else:
            return self.system_prompt

    # Optional rollout lifecycle management methods

    async def shutdown(self):
        pass

    async def init_rollout(self, rollout_id: str, **rollout_args) -> None:
        """Initialize resources for a new rollout"""
        return None

    async def release_rollout(self, rollout_id: str) -> None:
        """Free up resources for a new rollout. Called by compute_reward internally but also available for cleanup."""
        return None

    async def copy_to_workspace(
        self, rollout_id: str, src_path: Path, dst_filename: Optional[str] = None
    ) -> None:
        """Copy a file to the workspace for a specific rollout. If dst_filename is None, use the original filename."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support workspace file copy operations"
        )

    async def copy_content_to_workspace(
        self, rollout_id: str, src_content: str | bytes, dst_filename: str
    ) -> None:
        """Create a file with given content in the workspace for a specific rollout"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support workspace content copy operations"
        )

    async def copy_from_workspace(
        self, rollout_id: str, src_filename: str, dst_path: Path
    ) -> None:
        """Copy a file from the workspace for a specific rollout"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support workspace file retrieval operations"
        )


# Auto-wrap is applied via ``BaseEnv.__init_subclass__`` to SUBCLASS
# overrides only — not to ``BaseEnv``'s own defaults. The defaults here
# are either no-ops (``init_rollout`` / ``release_rollout``) or honest
# configuration errors (``copy_*`` raising NotImplementedError); wrapping
# them would only add cost without changing observability. If an env
# subclass overrides one of these, that override gets the wrap.
