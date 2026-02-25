__version__ = "2.0.0"

from .make import make

# Lazy — only import RocketSim if the legacy backend is requested.
# Users should access the backend through rlgym_sim.make(backend=...).

