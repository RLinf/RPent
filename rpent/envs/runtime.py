"""Lifecycle contract for environment-specific runtime adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rpent.tools.toolkit import Toolkit


class EnvRuntime(ABC):
    """Own the processes and clients needed by one RPent environment."""

    @abstractmethod
    def start(self) -> Toolkit:
        """Start or attach to the environment and return its agent toolkit."""

    @abstractmethod
    def stop(self) -> None:
        """Release environment processes and transport resources."""
