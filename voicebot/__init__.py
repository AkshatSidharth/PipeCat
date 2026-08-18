"""Pipecat voicebot with first-class observability.

Public surface:
    - :func:`voicebot.pipeline.build_pipeline` — assembles the bot pipeline.
    - :class:`voicebot.config.Settings` — environment-driven configuration.
"""

from voicebot.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
