"""variant_chain.sbatch is a valid, executable bash script."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts" / "slurm" / "variant_chain.sbatch"


def _site_roots():
    """The hygiene gate's generic roots plus any in HYGIENE_SITE_ROOTS."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_public_hygiene as hygiene

    env = {**hygiene.read_local_settings(ROOT), **os.environ}
    return tuple("/" + name for name in hygiene.site_roots(env))


def test_is_executable():
    assert SBATCH.stat().st_mode & 0o111


def test_bash_syntax_is_valid():
    result = subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_is_relocatable_and_logs_the_commit():
    text = SBATCH.read_text()
    assert 'cd "${SLURM_SUBMIT_DIR:?' in text
    assert "--output=logs/" in text and "--error=logs/" in text
    assert 'git -C "$SLURM_SUBMIT_DIR" rev-parse HEAD' in text
    # Built from parts so this file does not itself trip the path gate.
    for literal in ("/" + "home", "work" + "trees", "run_" + "2026", *_site_roots()):
        assert literal not in text, literal


def test_stage_scripts_exist():
    import re

    root = SBATCH.parents[2]
    for script in set(re.findall(r"scripts/[\w/]+\.py", SBATCH.read_text())):
        assert (root / script).is_file(), script


def test_log_directory_exists_in_a_fresh_clone():
    root = SBATCH.parents[2]
    assert (root / "logs" / ".gitkeep").is_file()
    assert "!logs/.gitkeep" in (root / ".gitignore").read_text().splitlines()
    assert "LOG_DIR" not in SBATCH.read_text()


def test_reports_tree_status_and_checks_the_repo_root():
    text = SBATCH.read_text()
    assert 'status --porcelain' in text
    assert "scripts/slurm/variant_chain.sbatch ] ||" in text
