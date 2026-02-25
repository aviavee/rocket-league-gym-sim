# rocket-league-gym-sim WORK IN PROGRESS

**vX.0.0** — A version of [RLGym](https://www.rlgym.org) for use with the [RocketSim](https://github.com/ZealanL/RocketSim) simulator.

Supports both the classic CPU-based C++ RocketSim backend and a new JAX GPU-accelerated backend, and is compatible with both the modern `gymnasium` API and the legacy `gym` API.

---

## Requirements

- Python ≥ 3.12
- numpy ≥ 2.0
- `gymnasium ≥ 1.0` (recommended) **or** `gym ≥ 0.17` (legacy)

Collision mesh assets from a copy of Rocket League you own are required for the simulator.  
Use the [RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper) to extract them and place the output folder at the top level of your project directory (`./collision_meshes/`).

---

## Installation

### Legacy backend (CPU, C++ RocketSim)
```bash
pip install "rlgym-sim[legacy]"
# or manually:
pip install rocketsim gymnasium
pip install git+https://github.com/aviavee/rocket-league-gym-sim@main
```

### JAX GPU backend
```bash
pip install "rlgym-sim[jax]"
```

### Core only (no simulator backend)
```bash
pip install "rlgym-sim[gymnasium]"
```

---

## Usage

`rlgym_sim` is a drop-in replacement for RLGym.  You can replace every `rlgym` import with `rlgym_sim` (or `import rlgym_sim as rlgym`) and existing code will work.

```python
import rlgym_sim

env = rlgym_sim.make(tick_skip=8, spawn_opponents=True)

obs, info = env.reset()          # gymnasium returns (obs, info)
done = False
while not done:
    actions = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
```

### gymnasium vs gym API

The step/reset signatures automatically adapt to whichever package is installed:

| Package | `reset()` | `step()` |
|---------|-----------|----------|
| `gymnasium` | `(obs, info)` | `(obs, reward, terminated, truncated, info)` |
| `gym` (legacy) | `obs` | `(obs, reward, done, info)` |

If both are installed, `gymnasium` takes priority.

### Selecting a backend

Pass `backend=` to `make()`:

```python
# Default — C++ RocketSim (CPU)
env = rlgym_sim.make(backend="legacy")

# JAX GPU backend (requires rlgym-sim[jax])
env = rlgym_sim.make(backend="jax", n_envs=1024)
```

### `make()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tick_skip` | `8` | Physics ticks per `step()` call |
| `spawn_opponents` | `False` | Include orange-team cars |
| `team_size` | `1` | Cars per team |
| `gravity` | `1.0` | Gravity scalar |
| `boost_consumption` | `1.0` | Boost drain rate scalar |
| `copy_gamestate_every_step` | `True` | Return a fresh `GameState` each step (slower but safe for cross-step comparisons) |
| `dodge_deadzone` | `0.8` | Min pitch magnitude required to trigger a dodge |
| `backend` | `"legacy"` | `"legacy"` (C++ RocketSim) or `"jax"` (JAX GPU) |
| `**backend_kw` | — | Extra kwargs forwarded to the backend (e.g. `n_envs=1024` for JAX) |

Client-specific options from the original RLGym (`use_injector`, `game_speed`, `auto_minimize`, etc.) are not applicable and have been removed.

---

## What's new in v2.0.0

- **gymnasium / gym compat shim** — auto-detects the installed package; step and reset signatures adapt accordingly.
- **Pluggable simulator backends** — abstract `SimBackend` interface with two implementations:
  - `LegacyBackend` — original C++ RocketSim (ZealanL / mtheall bindings, `pip install rocketsim`)
  - `JaxBackend` — GPU-accelerated JAX backend (BLMChoosen/RocketSim fork)
- **Lazy imports** — `import rlgym_sim` no longer hard-fails if `RocketSim` is not installed; the backend is only loaded when `make()` is called.
- **numpy ≥ 2.0** — updated throughout; dtype handling follows the numpy 2 spec.
- **Python ≥ 3.12** required.
- **Docker support** — `Dockerfile` (test image) and `Dockerfile.dev` (full training stack with RocketSim + PyTorch + SB3).
- **`__version__`** exposed via `rlgym_sim.__version__`.

---

## Known issues

- `PlayerData` does not yet track `match_saves` or `match_shots`.
- `SB3MultipleInstanceEnv` (from `rlgym_utils`) imports the `rlgym` library directly; replace those imports with `rlgym_sim` equivalents and remove the 60-second inter-launch delay (not needed with RocketSim).
- JAX backend mutator settings (gravity, boost consumption) are not yet wired through to the JAX RocketSim constants layer.

