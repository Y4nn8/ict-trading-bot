"""FeatureExtractor ABC and FeatureRegistry for the Midas engine.

Each extractor is a plugin that declares tunable parameters (for Optuna),
maintains internal state across ticks, and returns named float features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.midas.types import PartialCandle, Tick


@dataclass(frozen=True, slots=True)
class ExtractorParam:
    """A tunable parameter exposed to the Optuna outer loop.

    Args:
        name: Parameter name (local to the extractor).
        default: Default value.
        low: Lower bound for Optuna search.
        high: Upper bound for Optuna search.
        param_type: "float" or "int".
    """

    name: str
    default: float
    low: float
    high: float
    param_type: str = "float"


class FeatureExtractor(ABC):
    """Abstract base for Midas feature extractors.

    Lifecycle per replay run:
      1. configure(params) — apply tunable param values
      2. For each tick:
         a. If candle closed: on_candle_close(closed_candle, index)
         b. extract(tick, partial_candle, candle_index) → features
      3. reset() — between runs
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique extractor name, used as feature name prefix."""

    @abstractmethod
    def tunable_params(self) -> list[ExtractorParam]:
        """Declare tunable parameters for the Optuna outer loop."""

    @abstractmethod
    def configure(self, params: dict[str, float]) -> None:
        """Apply tunable param values before a replay run.

        Args:
            params: Mapping of local param name → value.
        """

    @abstractmethod
    def on_candle_close(
        self,
        closed_candle: dict[str, Any],
        candle_index: int,
    ) -> None:
        """Called when a candle closes. Update internal state.

        Args:
            closed_candle: Dict with time, open, high, low, close, tick_count.
            candle_index: Sequential index of the closed candle.
        """

    @abstractmethod
    def extract(
        self,
        tick: Tick,
        partial_candle: PartialCandle,
        candle_index: int,
    ) -> dict[str, float]:
        """Extract features for the current tick.

        Args:
            tick: The raw tick.
            partial_candle: The in-progress candle being built.
            candle_index: Index of the candle currently being built.

        Returns:
            Dict mapping feature_name → float. Names are prefixed
            with ``{self.name}__``.
        """

    def reset(self) -> None:  # noqa: B027
        """Reset internal state between replay runs.

        Override in subclasses that maintain state.
        """


class FeatureRegistry:
    """Composes multiple FeatureExtractors into a single pipeline.

    Handles param namespacing: ``{extractor.name}__{param.name}``.
    """

    def __init__(self) -> None:
        self._extractors: list[FeatureExtractor] = []

    def register(self, extractor: FeatureExtractor) -> None:
        """Add an extractor to the pipeline."""
        self._extractors.append(extractor)

    @property
    def extractors(self) -> list[FeatureExtractor]:
        """All registered extractors."""
        return list(self._extractors)

    def all_tunable_params(self) -> list[ExtractorParam]:
        """Collect all tunable params, namespaced by extractor.

        Returns:
            List of ExtractorParam with fully-qualified names.
        """
        result: list[ExtractorParam] = []
        for ext in self._extractors:
            for p in ext.tunable_params():
                result.append(
                    ExtractorParam(
                        name=f"{ext.name}__{p.name}",
                        default=p.default,
                        low=p.low,
                        high=p.high,
                        param_type=p.param_type,
                    ),
                )
        return result

    def configure_all(self, params: dict[str, float]) -> None:
        """Configure all extractors from a flat namespaced param dict.

        Args:
            params: Flat dict with keys like ``ict__lookback``.
                    Missing keys fall back to defaults.
        """
        for ext in self._extractors:
            local: dict[str, float] = {}
            for p in ext.tunable_params():
                key = f"{ext.name}__{p.name}"
                local[p.name] = params.get(key, p.default)
            ext.configure(local)

    def on_candle_close(
        self,
        closed_candle: dict[str, Any],
        candle_index: int,
    ) -> None:
        """Forward candle close to all extractors."""
        for ext in self._extractors:
            ext.on_candle_close(closed_candle, candle_index)

    def extract_all(
        self,
        tick: Tick,
        partial_candle: PartialCandle,
        candle_index: int,
    ) -> dict[str, float]:
        """Run all extractors and merge feature dicts.

        Args:
            tick: The raw tick.
            partial_candle: Current partial candle.
            candle_index: Index of the current candle.

        Returns:
            Merged dict of all features.
        """
        features: dict[str, float] = {}
        for ext in self._extractors:
            features.update(ext.extract(tick, partial_candle, candle_index))
        return features

    def reset_all(self) -> None:
        """Reset all extractors for a new replay run."""
        for ext in self._extractors:
            ext.reset()
