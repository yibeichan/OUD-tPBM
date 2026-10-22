"""The hypothesis-test modules import without the study environment, and share one hypothesis table."""
import os
import subprocess
import sys
from pathlib import Path

HYPO = Path(__file__).resolve().parents[1] / "scripts" / "hypo_test"

_CODE = """
import sys
sys.path.insert(0, {hypo!r})
import unified_ef_model_roi as roi, unified_ef_model_channel as ch, unified_ef_model_common as common
assert roi.PRIMARY_HYPOTHESES is common.PRIMARY_HYPOTHESES is ch.PRIMARY_HYPOTHESES
from utils import statistical_methods as sm
for loader in (sm.load_behavioral_data, sm.load_assessment_data):
    try:
        loader()
    except RuntimeError as exc:
        assert "BASE_DIR" in str(exc)
    else:
        raise SystemExit("loader did not raise")
print("ok")
"""


def test_import_without_env_and_loaders_raise():
    # Empty values stop load_dotenv from filling them in from a local .env.
    env = dict(os.environ, BASE_DIR="", OUTPUT_DIR="")
    result = subprocess.run(
        [sys.executable, "-c", _CODE.format(hypo=str(HYPO))],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("ok")
