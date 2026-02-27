from typing import Tuple

# Base seed can be overridden for reproducibility via MC_BASE_SEED.
# Needs to be in a callable module, TODO: Move to utils module
BASE_SEED = int(__import__("os").environ.get("MC_BASE_SEED", "20250227"))
SAMPLER_RNG = {"pid": None, "rng": None}

def sampler_A() -> Tuple[float, float, float]:
    """Parallel-safe normal sampler.

        Each process lazily initializes its own RNG using a SeedSequence that
        mixes a base seed with the process id, so forked workers do not reuse
        identical RNG state.
        """
    import os
    import numpy as np
    # HACK: Add base seed and sampler rng to a utils module
    from mc_methods import BASE_SEED, SAMPLER_RNG

    pid = os.getpid()
    if SAMPLER_RNG["pid"] != pid or SAMPLER_RNG["rng"] is None:
        seed_seq = np.random.SeedSequence([BASE_SEED, pid])
        SAMPLER_RNG["pid"] = pid
        SAMPLER_RNG["rng"] = np.random.default_rng(seed_seq)

    draws = SAMPLER_RNG["rng"].normal(loc=0.0, scale=1.0, size=3)
    return float(draws[0]), float(draws[1]), float(draws[2])

def sampler_B() -> Tuple[float, float]:
    """Parallel-safe normal sampler.

        Each process lazily initializes its own RNG using a SeedSequence that
        mixes a base seed with the process id, so forked workers do not reuse
        identical RNG state.
        """
    import os
    import numpy as np
    # HACK: Add base seed and sampler rng to a utils module
    from mc_methods import BASE_SEED, SAMPLER_RNG

    pid = os.getpid()
    if SAMPLER_RNG["pid"] != pid or SAMPLER_RNG["rng"] is None:
        seed_seq = np.random.SeedSequence([BASE_SEED, pid])
        SAMPLER_RNG["pid"] = pid
        SAMPLER_RNG["rng"] = np.random.default_rng(seed_seq)

    draws = SAMPLER_RNG["rng"].uniform(low=-1.0, high=1.0, size=3)
    return float(draws[0]), float(draws[1])
