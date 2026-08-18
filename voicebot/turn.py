"""Smart turn detection with a custom model and a tunable threshold.

Pipecat's ``LocalSmartTurnAnalyzerV3`` accepts a ``smart_turn_model_path``, so a
fine-tuned ONNX model drops straight in provided it keeps the v3 contract:

    input   input_features  (B, 80, 800) float32   whisper log-mel
    output  logits          (B, 1)       float32   **sigmoid probabilities**

What it does *not* expose is the decision threshold — ``_predict_endpoint``
hardcodes ``probability > 0.5``. That matters for a voice agent, because the two
errors are not equally expensive:

* calling a turn **complete** too early → the bot talks over the caller
* calling it **incomplete** too late → the bot is a beat slow to answer

The first is far worse, so a fine-tuned model's own operating point is usually
above 0.5. This module keeps all of v3's feature extraction and only re-derives
the decision from the probability, so raising the threshold costs nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

# What a v3-compatible model must look like.
_EXPECTED_INPUT = "input_features"
_EXPECTED_SHAPE = (80, 800)


class ThresholdedSmartTurnAnalyzer(LocalSmartTurnAnalyzerV3):
    """Smart turn v3 with a configurable end-of-turn threshold.

    ``threshold`` is the probability above which the turn is judged complete.
    Higher means the analyzer waits through more mid-turn pauses (fewer
    interruptions, slightly slower replies); lower means the opposite.
    """

    def __init__(self, *, threshold: float = 0.5, **kwargs):
        """Initialize the analyzer.

        Args:
            threshold: End-of-turn probability cutoff, 0-1.
            **kwargs: Passed to :class:`LocalSmartTurnAnalyzerV3`, including
                ``smart_turn_model_path``.
        """
        super().__init__(**kwargs)
        self._threshold = float(threshold)

    def _predict_endpoint(self, audio_array) -> dict[str, Any]:
        """Run v3's inference, then apply our own threshold to its probability."""
        result = super()._predict_endpoint(audio_array)
        # Reuses v3's log-mel extraction and ONNX session untouched; only the
        # complete/incomplete call changes.
        result["prediction"] = 1 if result["probability"] > self._threshold else 0
        return result


def validate_turn_model(path: str) -> str:
    """Check a custom turn model before the pipeline depends on it.

    Raises rather than falling back to the bundled model: silently reverting a
    Hindi deployment to the stock English model would be an invisible quality
    regression, which is worse than refusing to start.

    Args:
        path: Filesystem path to the ONNX model.

    Returns:
        A short human-readable description of the validated model.

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError: The model does not honour the v3 input/output contract.
    """
    model = Path(path).expanduser()
    if not model.is_file():
        raise FileNotFoundError(f"smart turn model not found: {model}")

    import onnxruntime as ort

    try:
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ValueError(f"could not load smart turn model {model.name}: {exc}") from exc

    inputs = {i.name: i for i in session.get_inputs()}
    if _EXPECTED_INPUT not in inputs:
        raise ValueError(
            f"{model.name} has inputs {list(inputs)}; smart turn v3 requires "
            f"'{_EXPECTED_INPUT}'"
        )

    # Shape is (batch, mels, frames) with a symbolic batch dimension.
    shape = tuple(inputs[_EXPECTED_INPUT].shape[1:])
    if shape != _EXPECTED_SHAPE:
        raise ValueError(
            f"{model.name} expects features {shape}; smart turn v3 requires "
            f"{_EXPECTED_SHAPE} (80 mel bins x 800 frames)"
        )

    size_mb = model.stat().st_size / 1_048_576
    return f"{model.name} ({size_mb:.0f}MB, {_EXPECTED_INPUT}{(-1,) + shape})"


def build_turn_analyzer(
    *, model_path: str, threshold: float, params
) -> LocalSmartTurnAnalyzerV3:
    """Build the turn analyzer, using a custom model when one is configured.

    Args:
        model_path: Path to a custom ONNX model, or "" for Pipecat's bundled one.
        threshold: End-of-turn probability cutoff.
        params: ``SmartTurnParams`` for the analyzer.
    """
    log = logger.bind(component="turn")

    if model_path.strip():
        described = validate_turn_model(model_path)  # raises on a bad model
        log.info(
            "using custom smart turn model",
            event="smart_turn_model",
            model=described,
            threshold=threshold,
        )
        return ThresholdedSmartTurnAnalyzer(
            threshold=threshold,
            smart_turn_model_path=str(Path(model_path).expanduser()),
            params=params,
        )

    log.info(
        "using bundled smart turn model",
        event="smart_turn_model",
        model="smart-turn-v3.2-cpu.onnx",
        threshold=threshold,
    )
    return ThresholdedSmartTurnAnalyzer(threshold=threshold, params=params)
