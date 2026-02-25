"""
Compatibility shim for gym / gymnasium.

Tries gymnasium first (modern API), falls back to gym (legacy).
Provides unified access to:
  - Env base class
  - spaces module
  - GYMNASIUM flag (True when gymnasium is the active backend)
"""

GYMNASIUM: bool = False

try:
    import gymnasium
    from gymnasium import Env as _Env
    from gymnasium import spaces
    GYMNASIUM = True
except ImportError:
    try:
        import gym
        from gym import Env as _Env
        from gym import spaces
    except ImportError:
        raise ImportError(
            "Neither 'gymnasium' nor 'gym' is installed. "
            "Install one of them:\n"
            "  pip install 'rlgym-sim[gymnasium]'   # recommended\n"
            "  pip install 'rlgym-sim[gym]'          # legacy"
        )

Env = _Env
