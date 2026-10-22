"""setup_run_tree.py builds one variant's run tree out of symlinks.

Every test works on tmp_path trees only -- no real run tree is touched.
``analysis_variant`` files are read from the real, committed ``config/analysis_variants/`` (safe: read-only, no I/O
side effects), so the copied ``analysis_variant.json`` is checked against
a real variant's content.
"""
import json
import subprocess
from pathlib import Path

import pytest

from scripts.setup_run_tree import SetupError, setup_run_tree

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIRS = ["01_assessments", "02_behav", "03_glm", "04_channel_mappings"]


def _make_source(tmp_path, name="source", extra_dirs=()):
    source = tmp_path / name
    for d in SHARED_DIRS:
        (source / d).mkdir(parents=True)
        (source / d / "marker.txt").write_text(d)
    for d in extra_dirs:
        (source / d).mkdir(parents=True)
    return source


def test_creates_expected_symlinks_and_variant_config(tmp_path):
    source = _make_source(tmp_path)
    dest = tmp_path / "main"

    setup_run_tree("main", source, dest)

    for d in SHARED_DIRS:
        link = dest / d
        assert link.is_symlink()
        assert link.resolve() == (source / d).resolve()
        assert (link / "marker.txt").read_text() == d

    variant_path = dest / "analysis_variant.json"
    assert variant_path.is_file()
    assert not variant_path.is_symlink()
    expected = json.loads((REPO_ROOT / "config" / "analysis_variants" / "main.json").read_text())
    assert json.loads(variant_path.read_text()) == expected


def test_main_variant_gets_no_session_info_symlink(tmp_path):
    source = _make_source(tmp_path)
    dest = tmp_path / "main"

    setup_run_tree("main", source, dest)

    assert not (dest / "04b_session_info").exists()


def test_refuses_nonempty_dest(tmp_path):
    source = _make_source(tmp_path)
    dest = tmp_path / "main"
    dest.mkdir()
    (dest / "stray_file.txt").write_text("already here")

    with pytest.raises(SetupError, match="non-empty"):
        setup_run_tree("main", source, dest)


def test_empty_existing_dest_is_fine(tmp_path):
    source = _make_source(tmp_path)
    dest = tmp_path / "main"
    dest.mkdir()

    setup_run_tree("main", source, dest)

    assert (dest / "analysis_variant.json").is_file()


def test_missing_source_dir_is_an_error(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for d in SHARED_DIRS[:-1]:
        (source / d).mkdir(parents=True)
    dest = tmp_path / "main"

    with pytest.raises(SetupError, match="04_channel_mappings"):
        setup_run_tree("main", source, dest)


def test_unknown_variant_is_an_error(tmp_path):
    source = _make_source(tmp_path)
    dest = tmp_path / "nope"

    with pytest.raises(SetupError, match="no analysis variant config"):
        setup_run_tree("nope", source, dest)


def test_non_main_variant_requires_sibling_main_tree(tmp_path):
    source = _make_source(tmp_path)
    dest = tmp_path / "supp_all_channels"

    with pytest.raises(SetupError, match="main not found"):
        setup_run_tree("supp_all_channels", source, dest)


def test_non_main_variant_requires_main_session_info(tmp_path):
    source = _make_source(tmp_path)
    main_tree = tmp_path / "main"
    main_tree.mkdir()
    dest = tmp_path / "supp_all_channels"

    with pytest.raises(SetupError, match="04b_session_info"):
        setup_run_tree("supp_all_channels", source, dest)


def test_non_main_variant_symlinks_session_info_from_sibling_main(tmp_path):
    source = _make_source(tmp_path)
    main_tree = tmp_path / "main"
    (main_tree / "04b_session_info").mkdir(parents=True)
    (main_tree / "04b_session_info" / "low_accuracy_sessions.csv").write_text("a,b\n1,2\n")
    dest = tmp_path / "supp_all_channels"

    setup_run_tree("supp_all_channels", source, dest)

    link = dest / "04b_session_info"
    assert link.is_symlink()
    assert link.resolve() == (main_tree / "04b_session_info").resolve()
    assert (link / "low_accuracy_sessions.csv").read_text() == "a,b\n1,2\n"


def test_link_stages_symlinks_matching_prefix_dirs(tmp_path):
    source = _make_source(tmp_path)
    main_tree = tmp_path / "main"
    (main_tree / "04b_session_info").mkdir(parents=True)
    for name in ("05_channel_group_analysis", "06_roi_activation", "07_roi_analysis",
                 "08_glm_topoplots", "09_hypothesis_testing"):
        (main_tree / name).mkdir()
        (main_tree / name / "marker.txt").write_text(name)
    dest = tmp_path / "supp_count"

    setup_run_tree(
        "supp_count", source, dest,
        link_from=main_tree, link_stages=["05", "06", "07", "08"],
    )

    for name in ("05_channel_group_analysis", "06_roi_activation",
                 "07_roi_analysis", "08_glm_topoplots"):
        link = dest / name
        assert link.is_symlink(), name
        assert link.resolve() == (main_tree / name).resolve()
    assert not (dest / "09_hypothesis_testing").exists()


def test_link_stage_10_links_only_roi_activations_alltimepoints(tmp_path):
    source = _make_source(tmp_path)
    main_tree = tmp_path / "main"
    (main_tree / "04b_session_info").mkdir(parents=True)
    longitudinal = main_tree / "10_longitudinal"
    (longitudinal / "roi_activations_alltimepoints" / "stroop").mkdir(parents=True)
    (longitudinal / "roi_activations_alltimepoints" / "stroop" / "data.csv").write_text("x\n1\n")
    (longitudinal / "unified_ef").mkdir()
    (longitudinal / "unified_ef" / "hypothesis_convergence.csv").write_text("y\n2\n")
    dest = tmp_path / "supp_count"

    setup_run_tree(
        "supp_count", source, dest,
        link_from=main_tree, link_stages=["10"],
    )

    dest_longitudinal = dest / "10_longitudinal"
    assert dest_longitudinal.is_dir()
    assert not dest_longitudinal.is_symlink()

    roi_link = dest_longitudinal / "roi_activations_alltimepoints"
    assert roi_link.is_symlink()
    assert roi_link.resolve() == (longitudinal / "roi_activations_alltimepoints").resolve()
    assert (roi_link / "stroop" / "data.csv").read_text() == "x\n1\n"

    assert not (dest_longitudinal / "unified_ef").exists()


def test_link_stages_requires_link_from(tmp_path):
    source = _make_source(tmp_path)
    main_tree = tmp_path / "main"
    (main_tree / "04b_session_info").mkdir(parents=True)
    dest = tmp_path / "supp_count"

    with pytest.raises(SetupError, match="link-from"):
        setup_run_tree("supp_count", source, dest, link_stages=["05"])


def test_missing_link_stage_dir_is_an_error(tmp_path):
    source = _make_source(tmp_path)
    main_tree = tmp_path / "main"
    (main_tree / "04b_session_info").mkdir(parents=True)
    dest = tmp_path / "supp_count"

    with pytest.raises(SetupError, match="'05_'"):
        setup_run_tree("supp_count", source, dest, link_from=main_tree, link_stages=["05"])


def test_readme_records_variant_source_links_branch_commit(tmp_path):
    source = _make_source(tmp_path)
    main_tree = tmp_path / "main"
    (main_tree / "04b_session_info").mkdir(parents=True)
    (main_tree / "05_channel_group_analysis").mkdir()
    dest = tmp_path / "supp_count"

    setup_run_tree(
        "supp_count", source, dest,
        link_from=main_tree, link_stages=["05"],
    )

    readme = (dest / "README.md").read_text()
    assert "variant: supp_count" in readme
    assert f"source: {source}" in readme
    assert f"link_from: {main_tree}" in readme
    assert "link_stages: 05" in readme

    def git(*args):
        # "unknown" outside a git checkout, as setup_run_tree records it
        done = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else "unknown"

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    commit = git("rev-parse", "HEAD")
    assert f"branch: {branch}" in readme
    assert f"commit: {commit}" in readme


def test_readme_records_none_when_no_link_from(tmp_path):
    source = _make_source(tmp_path)
    dest = tmp_path / "main"

    setup_run_tree("main", source, dest)

    readme = (dest / "README.md").read_text()
    assert "link_from: none" in readme
    assert "link_stages: none" in readme
