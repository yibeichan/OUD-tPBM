"""First-level GLM launch hygiene: OUTPUT_DIR precedence, protected trees, summary log.

None of these change a computed output; they cover the launcher, the guard and
the shared analysis_summary.json log.
"""
import json
import multiprocessing
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_BRAIN = _REPO / "scripts" / "brain"
sys.path.insert(0, str(_BRAIN))


def _write_exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.mark.parametrize("task", ["stroop", "wcst"])
@pytest.mark.parametrize("exported", [True, False])
def test_launcher_exported_output_dir_wins_over_env_file(tmp_path, task, exported):
    repo = tmp_path / "repo"
    (repo / "scripts" / "brain").mkdir(parents=True)
    (repo / "scripts" / "brain" / f"step1_glm_{task}.py").write_text("")
    (repo / ".env").write_text(f"BASE_DIR={tmp_path}/base\nOUTPUT_DIR={tmp_path}/from_env\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "seen.txt"
    _write_exe(bin_dir / "micromamba", "#!/bin/bash\nexit 0\n")
    _write_exe(bin_dir / "python", f'#!/bin/bash\necho "$OUTPUT_DIR" > {record}\n')

    env = {k: v for k, v in os.environ.items() if k not in ("OUTPUT_DIR", "SLURM_ARRAY_TASK_ID")}
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PROJECT_REPO"] = str(repo)
    if exported:
        env["OUTPUT_DIR"] = str(tmp_path / "from_caller")
    subprocess.run(["bash", str(_BRAIN / f"{task}_glm_1st.sh"), "--test"],
                   env=env, check=True, capture_output=True)
    expected = tmp_path / ("from_caller" if exported else "from_env")
    assert record.read_text().strip() == str(expected)


@pytest.mark.parametrize("task", ["stroop", "wcst"])
def test_step1_refuses_a_protected_output_dir(tmp_path, task):
    info = tmp_path / "info.csv"
    info.write_text("subj_id,task_id,filename\n")
    env = dict(os.environ, OUTPUT_DIR=str(tmp_path / "frozen_run" / "x"),
               BASE_DIR=str(tmp_path), PROTECTED_TREE_MARKERS="frozen_run")
    result = subprocess.run(
        [sys.executable, str(_BRAIN / f"step1_glm_{task}.py"), "--task_name", task,
         "--file_info", str(info), "--array_idx", "0"],
        env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "refusing to write" in result.stderr
    assert not (tmp_path / "frozen_run").exists()


def _write_one(args):
    log_dir, i = args
    from glm_summary import write_session_summary
    write_session_summary(log_dir, {"subject_id": f"S{i}", "task_id": f"t{i}", "status": "SUCCESS"})


def test_concurrent_sessions_all_land_in_the_shared_summary(tmp_path):
    log_dir = tmp_path / "logs" / "GLM" / "stroop"
    log_dir.mkdir(parents=True)
    n = 24
    with multiprocessing.get_context("spawn").Pool(8) as pool:
        pool.map(_write_one, [(str(log_dir), i) for i in range(n)])
    shared = json.loads((log_dir / "analysis_summary.json").read_text())
    assert sorted(e["subject_id"] for e in shared) == sorted(f"S{i}" for i in range(n))
    assert len(list((log_dir / "summaries").glob("*_summary.json"))) == n


def test_rerun_replaces_the_session_entry(tmp_path):
    from glm_summary import write_session_summary
    write_session_summary(tmp_path, {"subject_id": "S1", "task_id": "a", "status": "FAILED"})
    write_session_summary(tmp_path, {"subject_id": "S1", "task_id": "a", "status": "SUCCESS"})
    shared = json.loads((tmp_path / "analysis_summary.json").read_text())
    assert [e["status"] for e in shared] == ["SUCCESS"]


def test_lock_failure_does_not_raise(tmp_path, monkeypatch):
    import glm_summary

    def boom(*a, **k):
        raise OSError(37, "No locks available")

    monkeypatch.setattr(glm_summary.fcntl, "flock", boom)
    result = glm_summary.write_session_summary(
        tmp_path, {"subject_id": "S1", "task_id": "a", "status": "SUCCESS"})
    assert result is None
    assert (tmp_path / "summaries" / "S1_a_summary.json").is_file()


def test_corrupt_session_file_does_not_raise(tmp_path):
    from glm_summary import write_session_summary

    (tmp_path / "summaries").mkdir()
    (tmp_path / "summaries" / "S9_z_summary.json").write_text("{not json")
    assert write_session_summary(
        tmp_path, {"subject_id": "S1", "task_id": "a", "status": "SUCCESS"}) is None
    assert (tmp_path / "summaries" / "S1_a_summary.json").is_file()


def _run_step1_main_with_failing_summary(tmp_path, monkeypatch, fail_glm):
    import importlib.util

    spec = importlib.util.spec_from_file_location("step1_glm_stroop_x", _BRAIN / "step1_glm_stroop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    info = tmp_path / "info.csv"
    info.write_text("subj_id,task_id,filename\nS1,t1,f.snirf\n")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["x", "--task_name", "stroop", "--file_info", str(info), "--array_idx", "0"])

    class FakeAnalysis:
        contrast_maps = {"a": 1}
        used_noise_model = "ar1"
        design_matrix_rank = 3

        def __init__(self, *a):
            pass

        def process_subject(self, *a):
            if fail_glm:
                raise ValueError("glm exploded")

    monkeypatch.setattr(mod, "StroopfNIRSAnalysis", FakeAnalysis)
    monkeypatch.setattr(__import__("glm_summary").fcntl,
                        "flock", lambda *a, **k: (_ for _ in ()).throw(OSError(37, "no locks")))
    mod.main()
    return tmp_path / "out" / "logs" / "GLM" / "stroop"


def test_successful_glm_survives_a_summary_failure(tmp_path, monkeypatch):
    log_dir = _run_step1_main_with_failing_summary(tmp_path, monkeypatch, fail_glm=False)

    session_file = log_dir / "summaries" / "S1_t1_summary.json"
    assert session_file.is_file()
    assert json.loads(session_file.read_text())["status"] == "SUCCESS"
    assert not (log_dir / "analysis_summary.json").exists()


def test_glm_exception_is_not_masked_by_a_summary_failure(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="glm exploded"):
        _run_step1_main_with_failing_summary(tmp_path, monkeypatch, fail_glm=True)


@pytest.mark.parametrize("task", ["stroop", "wcst"])
def test_guard_passes_a_normal_output_dir(tmp_path, task):
    info = tmp_path / "info.csv"
    info.write_text("subj_id,task_id,filename\n")
    env = dict(os.environ, OUTPUT_DIR=str(tmp_path / "run_normal"), BASE_DIR=str(tmp_path))
    result = subprocess.run(
        [sys.executable, str(_BRAIN / f"step1_glm_{task}.py"), "--task_name", task,
         "--file_info", str(info), "--array_idx", "5"],
        env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "refusing to write" not in result.stderr
    assert "out of bounds" in result.stderr


def test_launcher_empty_exported_output_dir_does_not_override_env(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts" / "brain").mkdir(parents=True)
    (repo / "scripts" / "brain" / "step1_glm_stroop.py").write_text("")
    (repo / ".env").write_text(f"BASE_DIR={tmp_path}/b\nOUTPUT_DIR={tmp_path}/from_env\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "seen.txt"
    _write_exe(bin_dir / "micromamba", "#!/bin/bash\nexit 0\n")
    _write_exe(bin_dir / "python", f'#!/bin/bash\necho "$OUTPUT_DIR" > {record}\n')
    env = {k: v for k, v in os.environ.items() if k != "SLURM_ARRAY_TASK_ID"}
    env.update(PATH=f"{bin_dir}:{env['PATH']}", PROJECT_REPO=str(repo), OUTPUT_DIR="")
    subprocess.run(["bash", str(_BRAIN / "stroop_glm_1st.sh"), "--test"], env=env, check=True, capture_output=True)
    assert record.read_text().strip() == str(tmp_path / "from_env")


def _fake_launcher_env(tmp_path, task):
    """A fake checkout of the launcher plus stub micromamba/python/sbatch."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "brain").mkdir(parents=True)
    (repo / "scripts" / "brain" / f"step1_glm_{task}.py").write_text("")
    (repo / ".env").write_text(f"BASE_DIR={tmp_path}/base\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "seen.txt"
    _write_exe(bin_dir / "micromamba", "#!/bin/bash\nexit 0\n")
    _write_exe(bin_dir / "python", f'#!/bin/bash\necho "$1" > {record}\n')
    env = {k: v for k, v in os.environ.items()
           if k not in ("OUTPUT_DIR", "SLURM_ARRAY_TASK_ID", "PROJECT_REPO", "SLURM_SUBMIT_DIR")}
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return repo, env, record


@pytest.mark.parametrize("task", ["stroop", "wcst"])
def test_launcher_ignores_a_submit_dir_that_is_not_this_checkout(tmp_path, task):
    repo, env, record = _fake_launcher_env(tmp_path, task)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    env["SLURM_SUBMIT_DIR"] = str(elsewhere)
    launcher = repo / "scripts" / "brain" / f"{task}_glm_1st.sh"
    launcher.write_text((_BRAIN / f"{task}_glm_1st.sh").read_text())
    subprocess.run(["bash", str(launcher), "--test"], env=env, check=True, capture_output=True)
    assert record.read_text().strip() == str(repo / "scripts" / "brain" / f"step1_glm_{task}.py")


@pytest.mark.parametrize("task", ["stroop", "wcst"])
def test_launcher_uses_submit_dir_when_it_is_a_checkout(tmp_path, task):
    repo, env, record = _fake_launcher_env(tmp_path, task)
    env["SLURM_SUBMIT_DIR"] = str(repo)
    # Run the real launcher (not inside the fake repo), as a spooled script would be.
    subprocess.run(["bash", str(_BRAIN / f"{task}_glm_1st.sh"), "--test"],
                   env=env, check=True, capture_output=True)
    assert record.read_text().strip() == str(repo / "scripts" / "brain" / f"step1_glm_{task}.py")


@pytest.mark.parametrize("task", ["stroop", "wcst"])
def test_launcher_project_repo_still_wins(tmp_path, task):
    repo, env, record = _fake_launcher_env(tmp_path, task)
    env["PROJECT_REPO"] = str(repo)
    env["SLURM_SUBMIT_DIR"] = str(tmp_path)
    subprocess.run(["bash", str(_BRAIN / f"{task}_glm_1st.sh"), "--test"],
                   env=env, check=True, capture_output=True)
    assert record.read_text().strip() == str(repo / "scripts" / "brain" / f"step1_glm_{task}.py")


@pytest.mark.parametrize("task", ["stroop", "wcst"])
def test_launcher_logs_commit_and_tree_status(tmp_path, task):
    repo, env, _ = _fake_launcher_env(tmp_path, task)
    env["PROJECT_REPO"] = str(repo)
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True,
                                    capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    sha = git("rev-parse", "HEAD")
    result = subprocess.run(["bash", str(_BRAIN / f"{task}_glm_1st.sh"), "--test"],
                            env=env, check=True, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    assert lines[0] == f"commit: {sha}"
    assert lines[1].startswith("tree status (empty = clean):")
