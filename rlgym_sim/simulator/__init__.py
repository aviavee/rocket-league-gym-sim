from .sim_backend import SimBackend

# Re-export for backward compatibility; imports are lazy so they don't
# explode when RocketSim (C++) is not installed.
def _load_legacy():
    from .legacy_backend import LegacyBackend, Player
    return LegacyBackend, Player

# Keep the old name available — thin alias
def RocketSimGame(*args, **kwargs):
    """Backward-compatible factory — delegates to ``LegacyBackend``."""
    LegacyBackend, _ = _load_legacy()
    return LegacyBackend(*args, **kwargs)