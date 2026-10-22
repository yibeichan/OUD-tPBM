#!/usr/bin/env python3
"""Deterministic seeding for the inference stages.

Every permutation p-value and bootstrap confidence interval in this project is
a Monte Carlo estimate, and the exploratory channel-level tests are judged by
which channels survive FDR (corrected within hypothesis). A nudge
the size of Monte Carlo noise can therefore change *which* channels survive, so
seeding has to deliver two properties, not one:

*Reproducible* -- the same command twice gives the same numbers. One seeded
generator threaded through a run is enough for this.

*Decomposable* -- a given test gives the same number whether it ran alone,
first, or last. One advancing generator does **not** give this: each draw then
depends on how many draws preceded it, so ``--contrast X`` on its own does not
reproduce contrast X from inside the full loop, and inserting one extra test
ahead of two others shifts their p-values. ``rng_for_key`` gives it, by
deriving each test's generator from the test's *identity* rather than from its
position in the run.

This module is the single definition of the seed and of that derivation; every
stage imports from here, so the value cannot drift between stages.
"""
from __future__ import annotations

import zlib

import numpy as np

# Fixed default so a bare run is reproducible. Override with --seed.
DEFAULT_SEED = 20260922


def rng_for_key(key, seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return the generator for one test identity, independent of run order.

    ``key`` must identify the test -- a channel name, hypothesis id, contrast,
    ROI, or a tuple of those -- and must never be its position in a loop: an
    index is still sequence-coupled as soon as the set of channels changes.

    The digest is ``zlib.crc32`` rather than the builtin ``hash``, which is
    salted per process by PYTHONHASHSEED and would reintroduce exactly the
    nondeterminism this removes.
    """
    return np.random.default_rng([seed, zlib.crc32(str(key).encode())])


def _resolve_rng(rng=None):
    """Return ``rng``, or a generator on DEFAULT_SEED when none was passed.

    The fallback for leaf helpers called without a generator. Callers that run
    more than one test should use ``rng_for_key`` instead, so their tests do
    not all share one stream.
    """
    return np.random.default_rng(DEFAULT_SEED) if rng is None else rng
