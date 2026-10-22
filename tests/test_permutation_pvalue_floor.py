"""The permutation p-value estimator must never floor to an exact zero.

``p = mean(|r_perm| >= |r_obs|)`` returns exactly 0.0 whenever no permutation
is as extreme as the observation -- which happens routinely once the true
effect is large relative to n_perm. A literal 0.0 is not a valid p-value: it
claims infinite evidence from a finite number of draws, and it is
indistinguishable from "the estimator's floor" versus "an impossible event".
The standard fix is the add-one/Laplace continuity correction,
``p = (1 + count) / (1 + n_perm)``, which puts a floor at ``1 / (n_perm + 1)``
instead of zero.

This mattered concretely: at n_perm=1000, the channel-level hypothesis model
reported 5 FDR-significant channels that were exactly the 5 rows where the
biased estimator returned p=0.0 -- an artifact of the estimator's resolution,
not evidence.

Each test below forces the zero-exceedance case deterministically: an x/y
pair with a perfect observed correlation, and an rng whose ``permutation()``
always returns one fixed, less-extreme shuffle (never the identity or the
full reversal, so it can never tie or exceed the observed |r|=1.0).
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
_HYPO_DIR = _SCRIPTS / "hypo_test"
_BRAIN_DIR = _SCRIPTS / "brain"

N_PERM = 99
EXPECTED_FLOOR_P = 1.0 / (N_PERM + 1)


class _FixedPermGenerator:
    """A generator whose ``permutation()`` always returns one fixed shuffle.

    ``choice()`` (used for the bootstrap CI half of these functions) delegates
    to a real generator, since only the permutation draw needs to be pinned.
    """

    def __init__(self, fixed_permutation, real_rng):
        self._fixed = np.asarray(fixed_permutation)
        self._real = real_rng

    def permutation(self, arr):
        return self._fixed

    def choice(self, *args, **kwargs):
        return self._real.choice(*args, **kwargs)


def _zero_exceedance_rng():
    """An rng that can never produce a permuted |r| >= the perfect observed |r|=1.0."""
    y = np.arange(1, 11)
    fixed_perm = np.roll(y, 3)  # neither the identity nor the full reversal
    assert abs(np.corrcoef(y, fixed_perm)[0, 1]) < 1.0
    return _FixedPermGenerator(fixed_perm, np.random.default_rng(0))


class _FixedSignFlipGenerator:
    """A generator whose ``choice()`` always returns one fixed sign pattern.

    For ``one_sample_permutation_test``, which draws its null via
    ``rng.choice([-1, 1], size=n)`` rather than ``rng.permutation``.
    """

    def __init__(self, fixed_signs):
        self._fixed = np.asarray(fixed_signs)

    def choice(self, *args, **kwargs):
        return self._fixed


def _zero_exceedance_sign_flip_rng():
    """An rng whose one sign-flip pattern always gives a far-less-extreme |t|."""
    values = np.arange(1, 13)  # all positive -> large observed |t|
    alternating_signs = np.array([1, -1] * 6)
    return values, _FixedSignFlipGenerator(alternating_signs)


@pytest.fixture(scope="module")
def statistical_methods(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("output_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_HYPO_DIR))
        module = importlib.import_module("utils.statistical_methods")
        yield module
    for name in ("utils.statistical_methods", "utils", "seeding"):
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def common(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("output_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_HYPO_DIR))
        module = importlib.import_module("unified_ef_model_common")
        yield module
    for name in ("unified_ef_model_common", "utils.statistical_methods", "utils", "seeding"):
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def step3a(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("output_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_BRAIN_DIR))
        module = importlib.import_module("step3a_channel_group_analysis")
        yield module
    for name in ("step3a_channel_group_analysis", "roi_utils", "seeding"):
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def channel_stage(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("output_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_HYPO_DIR))
        module = importlib.import_module("unified_ef_model_channel")
        yield module
    for name in ("unified_ef_model_channel", "unified_ef_model_common",
                 "utils.statistical_methods", "utils", "seeding"):
        sys.modules.pop(name, None)


def test_statistical_methods_permutation_correlation_never_floors_to_zero(statistical_methods):
    x = np.arange(1, 11)
    y = np.arange(1, 11)
    rng = _zero_exceedance_rng()

    r_obs, p, ci_lower, ci_upper = statistical_methods.permutation_correlation(
        x, y, n_permutations=N_PERM, rng=rng
    )

    assert r_obs == pytest.approx(1.0)
    assert p > 0.0
    assert p == pytest.approx(EXPECTED_FLOOR_P)


def test_common_permutation_correlation_never_floors_to_zero(common):
    x = np.arange(1, 11)
    y = np.arange(1, 11)
    rng = _zero_exceedance_rng()

    p = common.permutation_correlation(x, y, n_perm=N_PERM, rng=rng)

    assert p > 0.0
    assert p == pytest.approx(EXPECTED_FLOOR_P)


def test_channel_test_correlation_fast_never_floors_to_zero(channel_stage):
    x = np.arange(1, 11)
    y = np.arange(1, 11)
    rng = _zero_exceedance_rng()

    result = channel_stage.test_correlation_fast(x, y, n_perm=N_PERM, n_boot=10, rng=rng)

    assert result["r"] == pytest.approx(1.0)
    assert result["p"] > 0.0
    assert result["p"] == pytest.approx(EXPECTED_FLOOR_P)


def test_step3a_one_sample_permutation_test_never_floors_to_zero(step3a):
    values, rng = _zero_exceedance_sign_flip_rng()

    result = step3a.one_sample_permutation_test(values, n_permutations=N_PERM, rng=rng)

    assert abs(result["t_statistic"]) > 1.0
    assert result["p_value"] > 0.0
    assert result["p_value"] == pytest.approx(EXPECTED_FLOOR_P)
