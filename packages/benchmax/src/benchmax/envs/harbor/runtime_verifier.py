from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harbor.models.trial.config import VerifierConfig

__all__ = ["RuntimeOnlyHarborVerifier"]


@dataclass(frozen=True, slots=True, repr=False, init=False)
class RuntimeOnlyHarborVerifier:
    """Keep a credential-bearing Harbor verifier out of serialized bundles.

    The wrapped config is copied at construction and remains opaque while it is
    retained by ``HarborEnv``. A fresh ordinary Harbor config is created only
    when BenchMax constructs a concrete trial for local execution.
    """

    _config: VerifierConfig

    def __init__(self, config: VerifierConfig) -> None:
        from harbor.models.trial.config import VerifierConfig

        if not isinstance(config, VerifierConfig):
            raise TypeError(
                "runtime-only Harbor verifier must wrap Harbor VerifierConfig, "
                f"got {type(config).__name__}"
            )
        object.__setattr__(self, "_config", config.model_copy(deep=True))

    @property
    def disabled(self) -> bool:
        """Whether the retained Harbor verifier is disabled."""

        return bool(self._config.disable)

    def _harbor_config(self) -> VerifierConfig:
        """Create the ordinary config consumed by one concrete Harbor trial."""

        return self._config.model_copy(deep=True)

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("runtime-only Harbor verifier configuration cannot be bundled")

    def __repr__(self) -> str:
        return "RuntimeOnlyHarborVerifier(<redacted>)"
