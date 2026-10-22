"""Permutation p-values and bootstrap CIs must be reproducible AND decomposable.

The assessment scripts already seed explicitly; the inference path used
global numpy state, so every re-run produced different p-values and CIs for
the same data -- including the ones quoted in the manuscript.

Seeding the run is necessary but not sufficient. One advancing generator makes
a run repeatable while leaving each test's draw dependent on how many draws
preceded it: ``--contrast X`` alone then disagrees with contrast X inside the
full loop, and inserting one test ahead of two others shifts their p-values.
Since the published findings are the few channels that clear FDR out of
thousands of tests, a nudge that size can change which channels survive. So the
stages key each test's generator on the test's identity (``rng_for_key``), and
the tests below pin that property as well as bare repeatability.

``unified_ef_model_common`` and ``utils.statistical_methods`` both call
``load_dotenv()`` and bind ``OUTPUT_DIR`` at import, and
``statistical_methods`` exits when it is unset. ``load_dotenv()`` does not
override a variable already present in the environment, so pointing
``BASE_DIR``/``OUTPUT_DIR`` at a temporary directory *before* the import is
what keeps these tests off whichever output root happens to be configured.
Nothing is written at import; the isolation is there so the test does not
silently bind to a real analysis tree.
"""
import ast
import importlib
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
_HYPO_DIR = _SCRIPTS / "hypo_test"
_SEEDING = _SCRIPTS / "seeding.py"

# Every stage that resamples, plus the longitudinal driver that shares the
# pipeline's invocation contract. Each must expose --seed on the same default.
ENTRY_POINTS = (
    _SCRIPTS / "brain" / "step3a_channel_group_analysis.py",
    _SCRIPTS / "brain" / "step3c_roi_group_analysis.py",
    _SCRIPTS / "hypo_test" / "unified_ef_model_roi.py",
    _SCRIPTS / "hypo_test" / "unified_ef_model_channel.py",
    _SCRIPTS / "longitudinal" / "unified_ef_longitudinal.py",
)

PINNED_SEED = 20260922

# The samplers that read numpy's global state. default_rng is deliberately
# absent: constructing a generator is the fix, not the problem.
GLOBAL_SAMPLER = re.compile(r"np\.random\.(permutation|shuffle|choice|randn|rand|randint|seed)\(")


@pytest.fixture(scope="module")
def common(tmp_path_factory):
    """``unified_ef_model_common``, imported against a throwaway output root."""
    tmp = tmp_path_factory.mktemp("output_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_HYPO_DIR))
        module = importlib.import_module("unified_ef_model_common")
        yield module
    # Drop only what this file imported, so the next test file does not
    # inherit a module bound to a temporary OUTPUT_DIR. All pure Python.
    for name in ("unified_ef_model_common", "utils.statistical_methods", "utils", "seeding"):
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def step3c(tmp_path_factory):
    """``step3c_roi_group_analysis``, imported against a throwaway output root."""
    tmp = tmp_path_factory.mktemp("output_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_SCRIPTS / "brain"))
        module = importlib.import_module("step3c_roi_group_analysis")
        yield module
    for name in ("step3c_roi_group_analysis", "roi_utils", "seeding"):
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def step3a(tmp_path_factory):
    """``step3a_channel_group_analysis``, imported against a throwaway root."""
    tmp = tmp_path_factory.mktemp("output_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_SCRIPTS / "brain"))
        module = importlib.import_module("step3a_channel_group_analysis")
        yield module
    for name in ("step3a_channel_group_analysis", "roi_utils", "seeding"):
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def hypothesis_stages(tmp_path_factory):
    """``unified_ef_model_roi`` and ``unified_ef_model_channel``, isolated."""
    tmp = tmp_path_factory.mktemp("output_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_HYPO_DIR))
        yield (
            importlib.import_module("unified_ef_model_roi"),
            importlib.import_module("unified_ef_model_channel"),
        )
    for name in ("unified_ef_model_roi", "unified_ef_model_channel",
                 "unified_ef_model_common", "utils.statistical_methods", "utils", "seeding"):
        sys.modules.pop(name, None)


def _cohort_frames(n_subjects=12):
    """A synthetic cohort shaped like what ``load_all_data`` returns."""
    import pandas as pd

    rng = np.random.default_rng(11)
    subjects = [f"S{i}" for i in range(1, n_subjects + 1)]
    merged = pd.DataFrame({
        "subject": subjects,
        "Inhibit_t": rng.normal(50, 10, n_subjects),
        "Shift_t": rng.normal(50, 10, n_subjects),
        "Working Memory_t": rng.normal(50, 10, n_subjects),
        "GEC_t": rng.normal(50, 10, n_subjects),
        "interference": rng.normal(size=n_subjects),
        "rt_variability": rng.normal(size=n_subjects),
        "perseverative_errors": rng.normal(size=n_subjects),
        "wm_maintenance_decline": rng.normal(size=n_subjects),
        "global_performance_z": rng.normal(size=n_subjects),
    })

    # Every ROI and contrast the five hypotheses reach for.
    roi = {}
    for task, contrasts in (("stroop", ["stroop_effect", "mixing_cost"]),
                            ("wcst", ["change_vs_regular", "learning"])):
        rows = []
        for contrast in contrasts:
            for roi_name in ("SalVentAttn_PFCl", "Cont_PFCl", "Cont_Par"):
                for subject in subjects:
                    rows.append({
                        "subject": subject, "contrast": contrast,
                        "network_roi": roi_name, "network": roi_name.split("_")[0],
                        "activation": rng.normal(), "t_statistic": rng.normal(),
                        "n_channels": 4,
                    })
        roi[task] = pd.DataFrame(rows)

    return merged, roi


def _channel_cache(roi_frames):
    """Pre-populated cache, so ``load_channel_data`` never reaches the disk.

    One channel in one ROI: ``test_correlation_fast`` hardcodes 1000
    permutations and 1000 bootstraps, so the cohort is kept to the smallest
    shape that still drives every hypothesis through the loop.
    """
    import pandas as pd

    rng = np.random.default_rng(12)
    cache = {}
    for task, frame in roi_frames.items():
        for contrast in frame["contrast"].unique():
            cache[f"{task}_{contrast}"] = pd.DataFrame([
                {"subject": subject, "channel": "S18_D8",
                 "activation": rng.normal(), "t_statistic": rng.normal(),
                 "p_value": 0.5, "network": "Cont", "network_roi": "Cont_PFCl"}
                for subject in frame["subject"].unique()
            ])
    return cache


def _spy_on_keys(monkeypatch, module):
    """Record every ``rng_for_key`` call the stage makes."""
    real = module.rng_for_key
    seen = []

    def spy(key, seed=PINNED_SEED):
        seen.append(key)
        return real(key, seed)

    monkeypatch.setattr(module, "rng_for_key", spy)
    return seen


def _p_values(result):
    return {test["test_name"]: test["p"] for test in result.get("tests", [])
            if test.get("status") == "completed"}


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=12)
    return x, x * 0.6 + rng.normal(size=12) * 0.5


def test_same_seed_gives_identical_p(common, data):
    x, y = data
    first = common.permutation_correlation(x, y, n_perm=200, rng=np.random.default_rng(7))
    second = common.permutation_correlation(x, y, n_perm=200, rng=np.random.default_rng(7))

    assert first == second


def test_different_seed_can_differ(common, data):
    x, y = data
    a = common.permutation_correlation(x, y, n_perm=200, rng=np.random.default_rng(1))
    b = common.permutation_correlation(x, y, n_perm=200, rng=np.random.default_rng(2))

    assert isinstance(a, float) and isinstance(b, float)


def test_default_seed_is_pinned(common):
    assert common.DEFAULT_SEED == PINNED_SEED


# =============================================================================
# DECOMPOSABILITY: a test's number must not depend on what ran before it
# =============================================================================

def test_p_value_is_the_same_alone_first_and_last(common, data):
    """Batch position must not move a p-value.

    This is the property a single advancing generator silently fails: with one
    shared stream, running "beta" after two other tests gives a different
    p-value than running it alone.
    """
    x, y = data

    def batch(keys):
        return {
            key: common.test_correlation(
                x, y, key, n_perm=100, n_boot=100,
                rng=common.rng_for_key(key, PINNED_SEED),
            )["p"]
            for key in keys
        }

    alone = batch(["beta"])["beta"]
    first = batch(["beta", "gamma", "delta"])["beta"]
    last = batch(["delta", "gamma", "beta"])["beta"]

    assert alone == first == last


def test_roi_bootstrap_ci_is_the_same_alone_and_in_the_full_frame(step3c):
    """The same property at stage level, on real stage code.

    ``compute_group_statistics`` loops over ROI x contrast cells; a shared
    generator would make each cell's CI depend on how many cells preceded it,
    so a re-run over a different ROI set would silently move the published CIs.
    """
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for roi in ("R1", "R2", "R3"):
        for subject in range(12):
            rows.append({
                "subject": f"S{subject}",
                "network_roi": roi,
                "network": "N",
                "contrast": "c",
                "activation": rng.normal(),
                "t_statistic": rng.normal(),
                "n_channels": 3,
            })
    full = pd.DataFrame(rows)
    subset = full[full["network_roi"] == "R2"].copy()

    from_full = step3c.compute_group_statistics(full, seed=PINNED_SEED)
    from_subset = step3c.compute_group_statistics(subset, seed=PINNED_SEED)

    r2_full = from_full[from_full["network_roi"] == "R2"].iloc[0]
    r2_subset = from_subset.iloc[0]

    assert (r2_full["mean_ci_lower"], r2_full["mean_ci_upper"]) == \
           (r2_subset["mean_ci_lower"], r2_subset["mean_ci_upper"])


def test_channel_statistics_do_not_depend_on_the_rest_of_the_run(step3a, monkeypatch):
    """The highest-stakes loop: which channels clear FDR is the published result.

    A channel's permutation p and bootstrap CI must be identical whether it was
    the only channel analysed or one of many, and whether or not the other
    contrasts ran in the same invocation -- otherwise ``--contrast X`` on its
    own disagrees with X inside ``main()``'s loop, and re-running after a
    channel is dropped silently moves every remaining channel's numbers.
    """
    import pandas as pd

    monkeypatch.setattr(step3a, "N_PERMUTATIONS", 50)
    monkeypatch.setattr(step3a, "N_BOOTSTRAP", 50)

    rng = np.random.default_rng(0)
    channels = ["S1_D1", "S18_D8", "S69_D97"]
    matrix = pd.DataFrame(
        {channel: rng.normal(loc=0.4, size=12) for channel in channels},
        index=[f"S{i}" for i in range(12)],
    )
    glm_data = pd.DataFrame([
        {"channel": channel, "network": "Default", "network_roi": "Default_PFC"}
        for channel in channels
    ])

    def stats_for(frame, contrast="learning"):
        out = step3a.run_channel_group_analysis(
            "wcst", contrast, frame, glm_data, 12, seed=PINNED_SEED
        )
        return out.set_index("channel")[["p_value", "d_ci_lower", "d_ci_upper"]]

    full = stats_for(matrix)
    alone = stats_for(matrix[["S18_D8"]])
    reordered = stats_for(matrix[["S69_D97", "S18_D8", "S1_D1"]])
    other_contrast = stats_for(matrix, contrast="change_vs_regular")

    assert full.loc["S18_D8"].to_dict() == alone.loc["S18_D8"].to_dict()
    assert full.loc["S18_D8"].to_dict() == reordered.loc["S18_D8"].to_dict()
    # Different contrast, same channel: a distinct key, so a distinct draw --
    # otherwise the contrasts' p-values would share one null and be coupled.
    assert full.loc["S18_D8"].to_dict() != other_contrast.loc["S18_D8"].to_dict()


def test_roi_hypothesis_p_values_do_not_depend_on_the_rest_of_the_sequence(
    hypothesis_stages, monkeypatch
):
    """The ROI stage's outputs are the confirmatory FDR family.

    Each hypothesis must give the same p-values whether its stage ran H3 alone
    or the whole H1->H5 sequence, and every resampled test within the run must
    draw from its own key -- two tests sharing a key share a null.
    """
    roi_stage, _ = hypothesis_stages
    merged, roi_frames = _cohort_frames()
    data = {"merged": merged, "roi_stroop": roi_frames["stroop"],
            "roi_wcst": roi_frames["wcst"]}

    # The stage calls test_correlation with its production 10,000/5,000 counts.
    # Shrinking them keeps the generator threading -- the property under test --
    # exactly as it is, and the run inside a test budget.
    real_test_correlation = roi_stage.test_correlation
    monkeypatch.setattr(
        roi_stage, "test_correlation",
        lambda x, y, name, n_perm=None, n_boot=None, *, rng=None:
            real_test_correlation(x, y, name, n_perm=100, n_boot=100, rng=rng),
    )

    keys = _spy_on_keys(monkeypatch, roi_stage)

    def run(hypo_ids):
        return {
            hypo_id: _p_values(roi_stage.test_hypothesis_roi_level(
                hypo_id, roi_stage.PRIMARY_HYPOTHESES[hypo_id], data, seed=PINNED_SEED
            ))
            for hypo_id in hypo_ids
        }

    everything = list(roi_stage.PRIMARY_HYPOTHESES)
    full = run(everything)
    keys_in_full = list(keys)
    keys.clear()

    alone = run(["H3"])
    reversed_order = run(list(reversed(everything)))

    assert full["H3"] == alone["H3"]
    assert full == reversed_order
    assert full["H3"], "H3 produced no completed tests -- the fixture is not exercising the stage"
    # Exactly one key per resampled test, never reused. Catches both a
    # collapsed key (two tests sharing a null) and a reverted shared generator
    # (no keys derived at all).
    assert len(keys_in_full) == sum(len(tests) for tests in full.values())
    assert len(keys_in_full) == len(set(keys_in_full))


def test_channel_hypothesis_p_values_do_not_depend_on_the_rest_of_the_sequence(
    hypothesis_stages, monkeypatch
):
    """Same property for the channel stage, whose FDR family is the exploratory one."""
    _, channel_stage = hypothesis_stages
    merged, roi_frames = _cohort_frames()
    # Empty ROI frames: the scatter-plot branch is not under test here.
    import pandas as pd
    data = {"merged": merged, "roi_stroop": pd.DataFrame(), "roi_wcst": pd.DataFrame()}

    # Same shrink as the ROI stage: the generator threading under test is
    # untouched, only the number of draws behind it.
    real_fast = channel_stage.test_correlation_fast
    monkeypatch.setattr(
        channel_stage, "test_correlation_fast",
        lambda x, y, n_perm=None, n_boot=None, *, rng=None:
            real_fast(x, y, n_perm=100, n_boot=100, rng=rng),
    )

    keys = _spy_on_keys(monkeypatch, channel_stage)

    def run(hypo_ids):
        cache = _channel_cache(roi_frames)
        return {
            hypo_id: _p_values(channel_stage.test_hypothesis_channel_level(
                hypo_id, channel_stage.PRIMARY_HYPOTHESES[hypo_id], data, cache,
                seed=PINNED_SEED,
            ))
            for hypo_id in hypo_ids
        }

    everything = list(channel_stage.PRIMARY_HYPOTHESES)
    full = run(everything)
    keys_in_full = list(keys)
    keys.clear()

    alone = run(["H2"])
    reversed_order = run(list(reversed(everything)))

    assert full["H2"] == alone["H2"]
    assert full == reversed_order
    assert full["H2"], "H2 produced no completed tests -- the fixture is not exercising the stage"
    assert len(keys_in_full) == sum(len(tests) for tests in full.values())
    assert len(keys_in_full) == len(set(keys_in_full))


def test_roi_bootstrap_ci_is_invariant_to_input_row_order(step3c):
    """Row order is not data.

    The bootstrap draws indices into ``roi_data['activation'].values``, whose
    order comes from step3b's concat -- i.e. from file-discovery order. Without
    a sort, the same seed on the same data yields a different published CI
    depending on how the rows happened to be stacked.
    """
    import pandas as pd

    rng = np.random.default_rng(3)
    frame = pd.DataFrame([
        {"subject": f"S{i}", "network_roi": "Cont_PFCl", "network": "Cont",
         "contrast": "learning", "activation": rng.normal(),
         "t_statistic": rng.normal(), "n_channels": 3}
        for i in range(12)
    ])
    shuffled = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert list(shuffled["subject"]) != list(frame["subject"])

    ordered = step3c.compute_group_statistics(frame, seed=PINNED_SEED).iloc[0]
    permuted = step3c.compute_group_statistics(shuffled, seed=PINNED_SEED).iloc[0]

    assert (ordered["mean_ci_lower"], ordered["mean_ci_upper"]) == \
           (permuted["mean_ci_lower"], permuted["mean_ci_upper"])


def test_key_derivation_survives_a_different_hash_seed():
    """The key digest must not be the builtin ``hash``.

    ``hash()`` on a string is salted per process by PYTHONHASHSEED, so keying
    on it would make every p-value depend on the interpreter's startup
    randomness -- reintroducing, invisibly, the nondeterminism being removed.
    """
    probe = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(_SCRIPTS)!r})
        from seeding import rng_for_key
        print(rng_for_key("S18_D8|learning").integers(0, 2**31, size=3).tolist())
        """
    )

    def run(hash_seed):
        return subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": hash_seed},
        ).stdout.strip()

    assert run("0") == run("12345")


# =============================================================================
# STATIC GUARDS: these catch a generator that stops being threaded
# =============================================================================

def _module_constants(tree):
    return {
        target.id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _seed_flag_default(path: Path) -> int:
    """Resolve the ``--seed`` default declared in ``path``'s argparse block.

    Parsed rather than imported: two of these modules pull in the GLM helpers
    and all of them exit when the environment is unset.

    A ``Name`` default is resolved against the file's own module-level
    constants and then against ``scripts/seeding.py``, which is where every
    stage's ``DEFAULT_SEED`` now comes from. There is deliberately no fallback
    to ``PINNED_SEED``: an unresolvable name must fail loudly rather than pass
    vacuously.
    """
    tree = ast.parse(path.read_text())
    constants = _module_constants(tree)
    constants.update({
        k: v for k, v in _module_constants(ast.parse(_SEEDING.read_text())).items()
        if k not in constants
    })

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        if not any(isinstance(a, ast.Constant) and a.value == "--seed" for a in node.args):
            continue
        for keyword in node.keywords:
            if keyword.arg != "default":
                continue
            if isinstance(keyword.value, ast.Constant):
                return keyword.value.value
            if isinstance(keyword.value, ast.Name):
                name = keyword.value.id
                assert name in constants, f"{path.name}: cannot resolve --seed default {name!r}"
                return constants[name]
            raise AssertionError(f"{path.name}: --seed default is not a name or literal")
        raise AssertionError(f"{path.name}: --seed declares no default")

    raise AssertionError(f"{path.name}: no --seed argument")


def test_every_stage_defaults_to_the_pinned_seed():
    """A stage that drifts off the shared default silently forks the results."""
    defaults = {path.name: _seed_flag_default(path) for path in ENTRY_POINTS}

    assert defaults == {path.name: PINNED_SEED for path in ENTRY_POINTS}


def _script_sources():
    for path in sorted(_SCRIPTS.rglob("*.py")):
        yield path, path.read_text()


def test_no_stage_samples_from_global_numpy_state():
    """The Step-5 grep, as a test.

    A new ``np.random.choice`` added anywhere under scripts/ ignores --seed
    entirely, and reads as reproducible because the surrounding code is.
    """
    offenders = [
        f"{path.relative_to(_REPO)}:{lineno}: {line.strip()}"
        for path, source in _script_sources()
        for lineno, line in enumerate(source.splitlines(), start=1)
        if GLOBAL_SAMPLER.search(line)
    ]

    assert offenders == []


def test_every_call_to_a_seeded_helper_passes_a_generator():
    """A helper that takes ``rng``/``seed`` must never be called without one.

    Dropping ``rng=`` from one call site is the quiet failure this catches:
    the helper falls back to a fresh generator on DEFAULT_SEED, so every
    channel draws from an identical stream and --seed is ignored for that
    statistic -- deterministic, plausible-looking, and wrong.

    Helpers are discovered by signature (a keyword-only ``rng`` or ``seed``
    parameter), not from a hardcoded list, so a helper added later is covered
    without touching this test.
    """
    seeded_params = {}
    trees = {}
    for path, source in _script_sources():
        trees[path] = tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            keyword_only = {a.arg for a in node.args.kwonlyargs} & {"rng", "seed"}
            if keyword_only:
                seeded_params.setdefault(node.name, set()).update(keyword_only)

    assert "test_correlation" in seeded_params, "discovery found no seeded helpers"

    offenders = []
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name not in seeded_params:
                continue
            passed = {k.arg for k in node.keywords}
            if not (passed & seeded_params[name]):
                expected = "/".join(sorted(seeded_params[name]))
                offenders.append(
                    f"{path.relative_to(_REPO)}:{node.lineno}: {name}() without {expected}="
                )

    assert offenders == []
