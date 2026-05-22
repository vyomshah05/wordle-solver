from .solver import FrequencySolver

__all__ = ["FrequencySolver", "FrequencyPlayer"]


def __getattr__(name: str):
    if name == "FrequencyPlayer":
        from .player import FrequencyPlayer

        return FrequencyPlayer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
