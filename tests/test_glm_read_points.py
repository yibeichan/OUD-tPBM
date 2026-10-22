"""The run's channel set, session exclusions and WCST masking at the GLM read points.

``read_glm_contrasts`` applies D1 (long channels only) and D6 (WCST sessions
without regular-incorrect trials have no maintenance contrasts);
``get_glm_with_roi_mappings`` applies D2 (sessions at or below chance
accuracy). Every downstream stage reads GLM output through these two, so
these tests pin what they return. Each test starts from a variant with all
three decisions off and turns on the one it tests.

``roi_utils`` binds ``OUTPUT_DIR``/``BASE_DIR`` at import, so the module is
imported with both pointed at a throwaway directory, and each test points
both the environment and the module values at its own tree.

Session ids here are synthetic, not real session hashes.
"""
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
_BRAIN = _REPO / "scripts" / "brain"
_HYPO = _REPO / "scripts" / "hypo_test"
_STAGE_MODULES = ("step3a_channel_group_analysis", "step3b_roi_extraction",
                  "unified_ef_model_common")

DECISIONS_OFF = {
    "name": "decisions_off",
    "channels": "all",
    "exclude_low_accuracy": False,
    "h2_metric": "perseverative_errors",
    "exclude_invalid_brief": False,
    "second_timepoint": "most_complete",
    "mask_wcst_without_regular_incorrect": False,
    "brain_direction": "efficiency",
}


@pytest.fixture(scope="module")
def roi_utils(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("import_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_BRAIN))
        mp.syspath_prepend(str(_HYPO))
        module = importlib.import_module("roi_utils")
        yield module
    # The stage modules close over this roi_utils; drop them with it.
    for name in ("roi_utils", *_STAGE_MODULES, "utils.statistical_methods", "utils", "seeding"):
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def stages(roi_utils, tmp_path_factory):
    """The three stages that read GLM output, imported against a throwaway root."""
    tmp = tmp_path_factory.mktemp("stage_import_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        yield {name: importlib.import_module(name) for name in _STAGE_MODULES}


@pytest.fixture
def tree(tmp_path, monkeypatch, roi_utils):
    """A fake OUTPUT_DIR/BASE_DIR pair, with env and roi_utils pointed at it."""
    out, base = tmp_path / "out", tmp_path / "base"
    out.mkdir()
    (base / "data").mkdir(parents=True)
    monkeypatch.setenv("OUTPUT_DIR", str(out))
    monkeypatch.setenv("BASE_DIR", str(base))
    monkeypatch.setattr(roi_utils, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(roi_utils, "BASE_DIR", str(base))
    return out, base


def write_variant(out, **overrides):
    (out / "analysis_variant.json").write_text(json.dumps({**DECISIONS_OFF, **overrides}))


def write_session_info(out):
    """aaaaaaa: S1_D2 short; bbbbbbb: S1_D1 short. Only aaaaaaa lacks regular-incorrect."""
    info = out / "04b_session_info"
    info.mkdir()
    short_rows = [
        ("S1", "aaaaaaa", 1, "S1_D1", 30.0, False),
        ("S1", "aaaaaaa", 1, "S1_D2", 8.0, True),
        ("S2", "bbbbbbb", 1, "S1_D1", 8.0, True),
        ("S2", "bbbbbbb", 1, "S1_D2", 30.0, False),
    ]
    cols = ["subject", "session_id", "task_order", "channel", "distance_mm", "is_short"]
    for task in ("stroop", "wcst"):
        pd.DataFrame(short_rows, columns=cols).to_csv(info / f"short_channels_{task}.csv", index=False)
    pd.DataFrame(
        [("S1", "aaaaaaa", 1, 10, 10, 50, 0), ("S2", "bbbbbbb", 1, 10, 10, 50, 3)],
        columns=["subject", "session_id", "task_order", "change", "post_change",
                 "regular_correct", "regular_incorrect"],
    ).to_csv(info / "wcst_condition_counts.csv", index=False)


def write_glm(path, contrasts, channels=((1, 1), (1, 2))):
    """One hbo row per (contrast, channel), in the column order step1 writes."""
    rows = []
    for contrast in contrasts:
        for i, (src, det) in enumerate(channels):
            rows.append({
                "effect": 0.5 + i, "stat": 1.5 + i, "p_value": 0.1, "se": 0.3, "df": 100.0,
                "ch_name": f"S{src}_D{det} hbo", "contrast": contrast, "chromophore": "hbo",
                "Source": src, "Detector": det,
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_long_channels_drop_only_this_sessions_short_channels(tree, roi_utils):
    out, _ = tree
    write_variant(out, channels="long")
    write_session_info(out)
    glm_a = write_glm(out / "03_glm" / "stroop" / "S1_aaaaaaa_glm_contrasts.csv", ["stroop_effect"])
    glm_b = write_glm(out / "03_glm" / "stroop" / "S2_bbbbbbb_glm_contrasts.csv", ["stroop_effect"])

    got_a = roi_utils.read_glm_contrasts(glm_a, "stroop", "aaaaaaa")
    got_b = roi_utils.read_glm_contrasts(glm_b, "stroop", "bbbbbbb")

    # S1_D2 is short in aaaaaaa only; S1_D1 is short in bbbbbbb only.
    assert got_a["channel"].tolist() == ["S1_D1"]
    assert got_b["channel"].tolist() == ["S1_D2"]
    assert got_a.index.tolist() == [0]


def test_wcst_mask_drops_maintenance_contrasts_for_sessions_without_regular_incorrect(tree, roi_utils):
    out, _ = tree
    write_session_info(out)
    contrasts = ["correct_vs_incorrect", "change_vs_regular", "learning", "duration_effect_correct"]
    glm_a = write_glm(out / "03_glm" / "wcst" / "S1_aaaaaaa_glm_contrasts.csv", contrasts)
    glm_b = write_glm(out / "03_glm" / "wcst" / "S2_bbbbbbb_glm_contrasts.csv", contrasts)

    write_variant(out, mask_wcst_without_regular_incorrect=True)
    masked = roi_utils.read_glm_contrasts(glm_a, "wcst", "aaaaaaa")
    assert masked["contrast"].tolist() == ["duration_effect_correct"] * 2
    # bbbbbbb has regular-incorrect trials, so nothing is masked.
    assert roi_utils.read_glm_contrasts(glm_b, "wcst", "bbbbbbb")["contrast"].tolist() == \
        [c for c in contrasts for _ in range(2)]

    write_variant(out, mask_wcst_without_regular_incorrect=False)
    unmasked = roi_utils.read_glm_contrasts(glm_a, "wcst", "aaaaaaa")
    assert unmasked["contrast"].tolist() == [c for c in contrasts for _ in range(2)]


def test_wcst_mask_raises_on_a_session_missing_from_condition_counts(tree, roi_utils):
    """A WCST session absent from wcst_condition_counts.csv must not be
    silently treated as having regular-incorrect trials (i.e. left unmasked)."""
    out, _ = tree
    write_session_info(out)
    counts_path = out / "04b_session_info" / "wcst_condition_counts.csv"
    df = pd.read_csv(counts_path)
    df[df.session_id != "aaaaaaa"].to_csv(counts_path, index=False)
    contrasts = ["correct_vs_incorrect", "change_vs_regular", "learning", "duration_effect_correct"]
    glm_a = write_glm(out / "03_glm" / "wcst" / "S1_aaaaaaa_glm_contrasts.csv", contrasts)

    write_variant(out, mask_wcst_without_regular_incorrect=True)
    with pytest.raises(KeyError, match="aaaaaaa"):
        roi_utils.read_glm_contrasts(glm_a, "wcst", "aaaaaaa")


def test_missing_variant_file_raises(tree, roi_utils):
    out, _ = tree
    glm = write_glm(out / "03_glm" / "stroop" / "S1_aaaaaaa_glm_contrasts.csv", ["stroop_effect"])
    with pytest.raises(Exception, match="analysis_variant.json"):
        roi_utils.read_glm_contrasts(glm, "stroop", "aaaaaaa")


def _pairing_tree(out, base):
    """Two baseline stroop sessions with ROI mappings; S2's accuracy is below chance."""
    for subj, sess in (("S1", "aaaaaaa"), ("S2", "bbbbbbb")):
        write_glm(out / "03_glm" / "stroop" / f"{subj}_{sess}_glm_contrasts.csv", ["stroop_effect"])
    mapping_dir = out / "04_channel_mappings"
    mapping_dir.mkdir()
    sessions = []
    for n, sess in (("01", "aaaaaaa"), ("02", "bbbbbbb")):
        name = f"roi_mapping_stroop_{n}_{sess}.json"
        (mapping_dir / name).write_text(json.dumps({"channel_to_network": {}}))
        sessions.append({"session_id": sess, "task": "stroop", "roi_mapping_file": name})
    (mapping_dir / "session_index.json").write_text(json.dumps({"sessions": sessions}))

    cols = ["filename", "subj_id", "task_id", "task_order"]
    pd.DataFrame([("a.snirf", "S1", "aaaaaaa", 1), ("b.snirf", "S2", "bbbbbbb", 1)],
                 columns=cols).to_csv(base / "data" / "stroop_task_info.csv", index=False)
    pd.DataFrame(columns=cols).to_csv(base / "data" / "wcst_task_info.csv", index=False)
    behav = out / "02_behav" / "stroop"
    behav.mkdir(parents=True)
    for subj, sess, acc in (("S1", "aaaaaaa", 0.9), ("S2", "bbbbbbb", 0.2)):
        (behav / f"{subj}_{sess}_results.json").write_text(
            json.dumps({"metrics": {"overall_accuracy": acc}}))


def test_get_glm_with_roi_mappings_skips_excluded_sessions(tree, roi_utils, caplog):
    out, base = tree
    _pairing_tree(out, base)

    write_variant(out, exclude_low_accuracy=True)
    with caplog.at_level("WARNING", logger=roi_utils.logger.name):
        kept = roi_utils.get_glm_with_roi_mappings("stroop", baseline_only=True)
    assert [(p[2], p[3]) for p in kept] == [("aaaaaaa", "S1")]
    skipped = [r.getMessage() for r in caplog.records
               if r.levelname == "WARNING" and "excluded" in r.getMessage()]
    assert len(skipped) == 1 and "S2" in skipped[0] and "bbbbbbb" in skipped[0]

    write_variant(out, exclude_low_accuracy=False)
    both = roi_utils.get_glm_with_roi_mappings("stroop", baseline_only=True)
    assert [(p[2], p[3]) for p in both] == [("aaaaaaa", "S1"), ("bbbbbbb", "S2")]


def _long_channel_tree(out, base):
    """The pairing tree, stroop, with S1_D2 short in aaaaaaa and S1_D1 short in bbbbbbb."""
    _pairing_tree(out, base)
    write_session_info(out)
    write_variant(out, channels="long")


def test_step3a_loader_reads_through_the_channel_rules(tree, roi_utils, stages):
    out, base = tree
    _long_channel_tree(out, base)
    step3a = stages["step3a_channel_group_analysis"]

    data = step3a.load_baseline_glm_data("stroop", "stroop_effect")

    assert sorted(zip(data["session_id"], data["channel"])) == [
        ("aaaaaaa", "S1_D1"), ("bbbbbbb", "S1_D2")]


def test_step3a_min_subjects_is_the_number_of_included_sessions(stages, monkeypatch):
    step3a = stages["step3a_channel_group_analysis"]
    monkeypatch.setattr(step3a, "N_PERMUTATIONS", 20)
    monkeypatch.setattr(step3a, "N_BOOTSTRAP", 20)
    matrix = pd.DataFrame({"S1_D1": [0.1, 0.2, 0.3], "S1_D2": [0.1, 0.2, float("nan")]},
                          index=["S1", "S2", "S3"])
    glm_data = pd.DataFrame({"channel": ["S1_D1", "S1_D2"], "network": ["Cont"] * 2,
                             "network_roi": ["Cont_PFCl"] * 2})

    three = step3a.run_channel_group_analysis("stroop", "stroop_effect", matrix, glm_data, 3)
    two = step3a.run_channel_group_analysis("stroop", "stroop_effect", matrix, glm_data, 2)

    assert three["channel"].tolist() == ["S1_D1"]
    assert two["channel"].tolist() == ["S1_D1", "S1_D2"]


def test_step3b_reads_through_the_channel_rules(tree, roi_utils, stages):
    out, base = tree
    _long_channel_tree(out, base)
    step3b = stages["step3b_roi_extraction"]
    glm = out / "03_glm" / "stroop" / "S2_bbbbbbb_glm_contrasts.csv"

    df = step3b.load_glm_results(glm, "stroop", "bbbbbbb")

    assert df["channel"].tolist() == ["S1_D2"]


def test_channel_hypothesis_loader_reads_through_the_channel_rules(tree, roi_utils, stages):
    out, base = tree
    _long_channel_tree(out, base)
    common = stages["unified_ef_model_common"]

    data = common.load_channel_data("stroop", "stroop_effect")

    assert sorted(zip(data["subject"], data["channel"])) == [("S1", "S1_D1"), ("S2", "S1_D2")]


def test_channel_hypothesis_loader_raises_on_a_missing_session_info_entry(tree, roi_utils, stages):
    """A session absent from short_channels_{task}.csv must not be silently skipped."""
    out, base = tree
    _long_channel_tree(out, base)
    info = out / "04b_session_info" / "short_channels_stroop.csv"
    df = pd.read_csv(info)
    df[df.session_id != "bbbbbbb"].to_csv(info, index=False)
    common = stages["unified_ef_model_common"]

    with pytest.raises(KeyError, match="bbbbbbb"):
        common.load_channel_data("stroop", "stroop_effect")


# --------------------------------------------------------------------------
# Session-set failures raise
# --------------------------------------------------------------------------

def test_a_session_without_roi_mapping_raises(tree, roi_utils):
    """A missing roi_mapping_*.json must not shrink the session set."""
    out, base = tree
    _pairing_tree(out, base)
    write_variant(out)
    index = out / "04_channel_mappings" / "session_index.json"
    payload = json.loads(index.read_text())
    payload["sessions"] = [s for s in payload["sessions"] if s["session_id"] != "bbbbbbb"]
    index.write_text(json.dumps(payload))
    (out / "04_channel_mappings" / "roi_mapping_stroop_02_bbbbbbb.json").unlink()

    with pytest.raises(FileNotFoundError, match="S2_bbbbbbb"):
        roi_utils.get_glm_with_roi_mappings("stroop", baseline_only=True)


def test_a_missing_session_index_raises(tree, roi_utils):
    """No session_index.json at all is the same failure, not an empty list."""
    out, base = tree
    _pairing_tree(out, base)
    write_variant(out)
    (out / "04_channel_mappings" / "session_index.json").unlink()
    for f in (out / "04_channel_mappings").glob("roi_mapping_*.json"):
        f.unlink()

    with pytest.raises(FileNotFoundError, match="No ROI mapping"):
        roi_utils.get_glm_with_roi_mappings("stroop", baseline_only=True)


def test_missing_task_info_raises(tree, roi_utils):
    """No task_info CSV means no baseline set; that is an error, not zero sessions."""
    out, base = tree
    _pairing_tree(out, base)
    write_variant(out)
    (base / "data" / "stroop_task_info.csv").unlink()

    with pytest.raises(FileNotFoundError, match="stroop_task_info.csv"):
        roi_utils.get_glm_with_roi_mappings("stroop", baseline_only=True)


def test_step3a_with_no_sessions_exits_nonzero(tree, roi_utils, stages, monkeypatch):
    """step3a must not exit 0 with an empty 05_channel_group_analysis."""
    out, base = tree
    _pairing_tree(out, base)
    write_variant(out)
    cols = ["filename", "subj_id", "task_id", "task_order"]
    pd.DataFrame(columns=cols).to_csv(base / "data" / "stroop_task_info.csv", index=False)
    step3a = stages["step3a_channel_group_analysis"]
    monkeypatch.setattr(step3a, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(sys, "argv", ["step3a", "--task", "stroop"])

    with pytest.raises(SystemExit) as exc:
        step3a.main()
    assert exc.value.code not in (0, None)


def test_every_lookup_reads_the_same_tree(tree, roi_utils, tmp_path):
    """GLM files and mappings follow OUTPUT_DIR at call time, like the variant does."""
    out, base = tree
    _pairing_tree(out, base)
    write_variant(out)
    stale = tmp_path / "stale_import_time_tree"
    (stale / "03_glm" / "stroop").mkdir(parents=True)
    roi_utils.OUTPUT_DIR = str(stale)  # restored by the tree fixture's monkeypatch
    roi_utils.BASE_DIR = str(stale)

    pairs = roi_utils.get_glm_with_roi_mappings("stroop", baseline_only=True)

    assert [(p[2], p[3]) for p in pairs] == [("aaaaaaa", "S1"), ("bbbbbbb", "S2")]
    assert all(str(p[0]).startswith(str(out)) and str(p[1]).startswith(str(out)) for p in pairs)


def test_step3a_raises_when_a_session_lacks_the_contrast(tree, roi_utils, stages):
    """A masked contrast in one baseline session must not silently remove it."""
    out, base = tree
    _pairing_tree(out, base)
    write_variant(out)
    write_glm(out / "03_glm" / "stroop" / "S2_bbbbbbb_glm_contrasts.csv", ["mixing_cost"])
    step3a = stages["step3a_channel_group_analysis"]

    with pytest.raises(ValueError, match="stroop_effect.*S2_bbbbbbb"):
        step3a.load_baseline_glm_data("stroop", "stroop_effect")


def _roi_file(out, task, rows):
    path = out / "06_roi_activation" / task / "all_subjects_roi_activation_network_roi.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["subject", "session_id", "contrast", "network_roi",
                                "activation", "t_statistic", "n_channels"]).to_csv(path, index=False)
    return path


def test_roi_hypothesis_loader_rejects_several_sessions_per_subject(tree, stages, monkeypatch):
    """An all-sessions step3b file must not reach the confirmatory tests."""
    out, _ = tree
    common = stages["unified_ef_model_common"]
    monkeypatch.setattr(common, "OUTPUT_DIR", str(out))
    _roi_file(out, "stroop", [
        ("S1", "aaaaaaa", "stroop_effect", "Cont_PFCl", 0.1, 1.0, 5),
        ("S1", "ccccccc", "stroop_effect", "Cont_PFCl", 0.2, 1.0, 5),
    ])
    with pytest.raises(ValueError, match="more than one session"):
        common._load_roi_activation("stroop")


def test_roi_hypothesis_loader_raises_on_a_missing_file(tree, stages, monkeypatch):
    """A missing ROI file must not drop the task's hypotheses from the family."""
    out, _ = tree
    common = stages["unified_ef_model_common"]
    monkeypatch.setattr(common, "OUTPUT_DIR", str(out))
    with pytest.raises(FileNotFoundError, match="06_roi_activation"):
        common._load_roi_activation("wcst")


def test_roi_hypothesis_loader_accepts_one_session_per_subject(tree, stages, monkeypatch):
    out, _ = tree
    common = stages["unified_ef_model_common"]
    monkeypatch.setattr(common, "OUTPUT_DIR", str(out))
    _roi_file(out, "stroop", [
        ("S1", "aaaaaaa", "stroop_effect", "Cont_PFCl", 0.1, 1.0, 5),
        ("S2", "bbbbbbb", "stroop_effect", "Cont_PFCl", 0.2, 1.0, 5),
    ])
    assert len(common._load_roi_activation("stroop")) == 2


def test_channel_hypothesis_loader_raises_with_no_sessions(tree, roi_utils, stages):
    """No baseline sessions must not silently drop the task's hypotheses."""
    out, base = tree
    _long_channel_tree(out, base)
    cols = ["filename", "subj_id", "task_id", "task_order"]
    pd.DataFrame(columns=cols).to_csv(base / "data" / "stroop_task_info.csv", index=False)
    with pytest.raises(FileNotFoundError, match="task=stroop"):
        stages["unified_ef_model_common"].load_channel_data("stroop", "stroop_effect")


def test_channel_hypothesis_loader_raises_on_an_unreadable_csv(tree, roi_utils, stages):
    """A truncated GLM CSV must not remove a participant from every channel test."""
    out, base = tree
    _long_channel_tree(out, base)
    (out / "03_glm" / "stroop" / "S2_bbbbbbb_glm_contrasts.csv").write_text("")
    with pytest.raises(pd.errors.EmptyDataError):
        stages["unified_ef_model_common"].load_channel_data("stroop", "stroop_effect")


def test_channel_hypothesis_loader_warns_on_a_session_without_the_contrast(
        tree, roi_utils, stages, caplog):
    """A session without the contrast is left out with a warning, not silently."""
    out, base = tree
    _long_channel_tree(out, base)
    write_glm(out / "03_glm" / "stroop" / "S2_bbbbbbb_glm_contrasts.csv", ["mixing_cost"])
    common = stages["unified_ef_model_common"]
    with caplog.at_level("WARNING", logger=common.logger.name):
        data = common.load_channel_data("stroop", "stroop_effect")
    assert sorted(set(data["subject"])) == ["S1"]
    assert any("bbbbbbb" in r.getMessage() and "stroop_effect" in r.getMessage()
               for r in caplog.records)
