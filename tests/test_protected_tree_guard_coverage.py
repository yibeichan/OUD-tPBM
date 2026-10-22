"""Every stage entry point that writes into OUTPUT_DIR refuses a protected run tree.

The static check parses each script: ``main()`` must have the
``refuse_protected_tree`` call as a top-level statement (bare, or the sole
statement of a ``try`` that turns the RuntimeError into an exit), and only
argument parsing, environment reads, ``print``, exits and path resolution may
run before it. A guard nested in an ``if``, or placed after a directory
creation, fails. The module body outside functions must not write at all, so
importing a stage cannot create directories either. The runtime check runs two of the stages against a
protected OUTPUT_DIR and confirms they exit without creating anything.
"""
import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
from analysis_variant import PROTECTED_TREE_MARKERS_VAR  # noqa: E402

GUARDED_ENTRY_POINTS = [
    "assessments/00_load_assessment_data.py",
    "assessments/01_analyze_executive_function.py",
    "assessments/01_plot_executive_function_t-score.py",
    "assessments/02_analyze_phq9.py",
    "assessments/03_analyze_gad7.py",
    "assessments/04_analyze_craving.py",
    "assessments/05_analyze_tea.py",
    "assessments/06_longitudinal_comparison.py",
    "behav/01_stroop_behav.py",
    "behav/01_wcst_behav.py",
    "brain/step1_glm_stroop.py",
    "brain/step1_glm_wcst.py",
    "brain/step2a_glm_to_parcellation.py",
    "brain/step2b_parcellation_to_roi.py",
    "brain/step3a_channel_group_analysis.py",
    "brain/step3b_roi_extraction.py",
    "brain/step3c_roi_group_analysis.py",
    "brain/step4_glm_topoplots.py",
    "hypo_test/unified_ef_model_roi.py",
    "hypo_test/unified_ef_model_channel.py",
    "longitudinal/extract_roi_alltimepoints.py",
    "longitudinal/longitudinal_exploratory.py",
    "longitudinal/unified_ef_longitudinal.py",
]

WRITE_ATTRS = {"mkdir", "makedirs", "to_csv", "to_json", "to_excel", "savefig",
               "write_text", "write_bytes", "dump", "save"}
WRITE_NAMES = {"open"}


def _writes(node: ast.AST) -> list[str]:
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Attribute) and fn.attr in WRITE_ATTRS:
                found.append(fn.attr)
            elif isinstance(fn, ast.Name) and fn.id in WRITE_NAMES:
                found.append(fn.id)
    return found


ALLOWED_BARE_CALLS = {
    "print", "exit", "parse_args", "Path", "str", "all", "any", "bool",
    "load_dotenv", "ArgumentParser", "SystemExit", "ValueError", "RuntimeError",
}
ALLOWED_ATTR_CALLS = {
    "parse_args", "add_argument", "add_argument_group",
    "add_mutually_exclusive_group", "ArgumentParser", "getenv", "exit",
}


def _is_guard_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "refuse_protected_tree")


def _is_guard_statement(stmt: ast.stmt) -> bool:
    """A bare guard call, or a try whose body is only the guard call."""
    if isinstance(stmt, ast.Expr):
        return _is_guard_call(stmt.value)
    if isinstance(stmt, ast.Try):
        return (len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Expr)
                and _is_guard_call(stmt.body[0].value))
    return False


def _disallowed_calls(node: ast.AST) -> list[str]:
    """Calls under ``node`` that are not on the pre-guard allowlist."""
    bad = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if isinstance(fn, ast.Name):
            ok = fn.id in ALLOWED_BARE_CALLS or fn.id.startswith("resolve_")
            name = fn.id
        elif isinstance(fn, ast.Attribute):
            ok = (fn.attr in ALLOWED_ATTR_CALLS or fn.attr.startswith("resolve_")
                  or (fn.attr == "get" and ast.unparse(fn.value) == "os.environ"))
            name = ast.unparse(fn)
        else:
            ok, name = False, ast.unparse(fn)
        if not ok:
            bad.append(name)
    return bad


def _first_disallowed_before_guard(main_fn: ast.FunctionDef):
    """Return (guard_index, disallowed calls before it); index is None if no top-level guard."""
    body = main_fn.body
    guard_at = next((i for i, stmt in enumerate(body) if _is_guard_statement(stmt)), None)
    if guard_at is None:
        return None, []
    return guard_at, [c for stmt in body[:guard_at] for c in _disallowed_calls(stmt)]


def _module_level_statements(tree: ast.Module):
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if (isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare)
                and isinstance(stmt.test.left, ast.Name)
                and stmt.test.left.id == "__name__"):
            continue
        yield stmt


@pytest.mark.parametrize("rel", GUARDED_ENTRY_POINTS)
def test_main_calls_guard_before_any_write(rel):
    tree = ast.parse((SCRIPTS / rel).read_text())
    mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert len(mains) == 1, f"{rel}: expected one top-level main()"
    guard_at, before = _first_disallowed_before_guard(mains[0])
    assert guard_at is not None, (
        f"{rel}: main() has no top-level refuse_protected_tree statement")
    assert not before, f"{rel}: main() runs {before} before the guard"


def test_a_guard_after_a_write_or_inside_an_if_is_flagged():
    def check(src):
        fn = ast.parse(src).body[0]
        return _first_disallowed_before_guard(fn)

    ok = "def main():\n    out = os.getenv('X')\n    refuse_protected_tree(out)\n"
    assert check(ok) == (1, [])
    late = ("def main():\n    ensure_output_directory(out)\n"
            "    refuse_protected_tree(out)\n")
    assert check(late)[1] == ["ensure_output_directory"]
    nested = "def main():\n    if x:\n        refuse_protected_tree(out)\n"
    assert check(nested)[0] is None
    wide_try = ("def main():\n    try:\n        refuse_protected_tree(out)\n"
                "        out.mkdir()\n    except RuntimeError:\n        pass\n")
    assert check(wide_try)[0] is None


@pytest.mark.parametrize("rel", GUARDED_ENTRY_POINTS)
def test_module_body_does_not_write(rel):
    tree = ast.parse((SCRIPTS / rel).read_text())
    writes = [w for stmt in _module_level_statements(tree) for w in _writes(stmt)]
    assert not writes, f"{rel}: writes at import time ({writes})"


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rel", [
    "assessments/00_load_assessment_data.py",
    "assessments/02_analyze_phq9.py",
])
def test_stage_exits_on_a_protected_output_dir(rel, tmp_path, monkeypatch):
    protected = tmp_path / "frozen_reference" / "main"
    monkeypatch.setenv(PROTECTED_TREE_MARKERS_VAR, "frozen_reference")
    monkeypatch.setenv("BASE_DIR", str(tmp_path / "base"))
    monkeypatch.setenv("OUTPUT_DIR", str(protected))
    module = _load(rel, "guard_probe_" + Path(rel).stem)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert "frozen_reference" in str(exc.value.code)
    assert not protected.exists()


def _longitudinal_comparison(tmp_path, monkeypatch):
    protected = tmp_path / "frozen_reference" / "main"
    monkeypatch.setenv(PROTECTED_TREE_MARKERS_VAR, "frozen_reference")
    monkeypatch.setenv("BASE_DIR", str(tmp_path / "base"))
    monkeypatch.setenv("OUTPUT_DIR", str(protected))
    module = _load("assessments/06_longitudinal_comparison.py", "guard_probe_longitudinal")
    return module, protected


def test_longitudinal_comparison_refuses_protected_default_output(tmp_path, monkeypatch):
    module, protected = _longitudinal_comparison(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["06_longitudinal_comparison.py"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert "frozen_reference" in str(exc.value.code)
    assert not protected.exists()


def test_longitudinal_comparison_with_output_dir_does_not_create_scores_dir(
        tmp_path, monkeypatch):
    module, protected = _longitudinal_comparison(tmp_path, monkeypatch)
    safe = tmp_path / "safe_out"
    monkeypatch.setattr(sys, "argv", ["06_longitudinal_comparison.py",
                                      "--output-dir", str(safe)])
    try:
        module.main()
    except (SystemExit, Exception):
        pass  # the inputs are absent; only where main() has written matters
    assert not protected.exists()
    assert safe.exists()
