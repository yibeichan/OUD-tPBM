"""The checker must catch each rule class and stay quiet on clean prose.

Every pattern and every filter is exercised behaviourally: each test below
fails when the thing it covers is deleted from the checker. Asserting that a
name appears in ``EXEMPT`` does not do that -- such a test passes with the
filter that reads ``EXEMPT`` removed -- so the exemptions are tested through
what the checker does with them.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_public_hygiene as mod
from check_public_hygiene import (
    Violation,
    check_tracked_floor,
    scan_content,
    scan_filenames,
    tracked_text_files,
)


# The settings a deployment keeps in its untracked .env, read before any
# fixture replaces them, so the real-tree test below can check against them.
SETTING_NAMES = ("HYGIENE_SITE_ROOTS", "HYGIENE_ID_PATTERNS")
_REAL_SETTINGS = {k: os.environ[k] for k in SETTING_NAMES if k in os.environ}

# Synthetic stand-ins for one site's settings. Every test runs with these
# set (see ``site_settings``); a test of the unset case removes them.
CONFIGURED_ROOTS = ("sitex", "clusterfs")
ID_PREFIX = "RAWID"

# The generic roots, plus the configured ones, pinned here independently of
# the checker's own tuple. Parametrising over ``mod.site_roots()`` alone
# would be self-referential: deleting a root from it deletes the very test
# case that should have caught the deletion, so the mutation passes.
GENERIC_ROOTS = ("Users",)
SITE_ROOTS = GENERIC_ROOTS + CONFIGURED_ROOTS

HOME_PATTERN = r"/home/[A-Za-z0-9][A-Za-z0-9_-]*"


@pytest.fixture(autouse=True)
def site_settings(monkeypatch):
    """Replace whatever the shell or a loaded .env set with synthetic values."""
    monkeypatch.setenv("HYGIENE_SITE_ROOTS", ",".join(CONFIGURED_ROOTS))
    monkeypatch.setenv("HYGIENE_ID_PATTERNS", ID_PREFIX)


@pytest.fixture
def unset_settings(monkeypatch):
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repo the checker can scan."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)

    def write(rel, text):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            path.write_bytes(text)
        else:
            path.write_text(text)
        subprocess.run(["git", "-C", str(tmp_path), "add", rel], check=True)
        return rel

    return write


@pytest.fixture
def run_main(repo, monkeypatch, capsys):
    """Run ``main()`` over a throwaway repo padded past the tracked floor.

    The padding matters: without it every ``main()`` test would trip the
    floor and stop distinguishing the rule under test from that one.
    """

    def go(*, pad=True):
        monkeypatch.delenv("BASE_DIR", raising=False)
        monkeypatch.delenv("OUD_TPBM_BASE_DIR", raising=False)
        if pad:
            for n in range(mod.MIN_TRACKED_FILES):
                repo(f"filler/f{n}.txt", f"clean line {n}\n")
        monkeypatch.setattr(sys, "argv", ["check_public_hygiene.py"])
        code = mod.main()
        return code, capsys.readouterr().out

    return go


# --- rule class 1: absolute paths ------------------------------------------


def test_flags_absolute_site_paths(repo):
    rel = repo("run.sh", 'DATA=/sitex/data/example/users/someone/x\n')
    found = scan_content([rel])
    assert [v.rule for v in found] == ["absolute-path"]
    assert found[0].line == 1


def test_flags_home_directory_paths(repo):
    rel = repo("run.sh", 'LOG=/home/someone/logs\n')
    assert [v.rule for v in scan_content([rel])] == ["absolute-path"]


def test_flags_a_second_configured_root(repo):
    rel = repo("run.sh", 'ARCHIVE=/clusterfs/group/x\n')
    assert [v.rule for v in scan_content([rel])] == ["absolute-path"]


@pytest.mark.parametrize("root", SITE_ROOTS)
def test_flags_every_site_root_written_without_a_trailing_slash(repo, root):
    """A root needs no trailing slash to disclose the site.

    Parametrised over this file's own ``SITE_ROOTS``, not the checker's, so
    a root dropped from the checker's tuple fails here instead of silently
    removing its own test case. Roots once kept a bare trailing-slash form
    and missed both this spelling and the next test's.
    """
    rel = repo("run.sh", f'BASE="/{root}"\n')
    assert [v.rule for v in scan_content([rel])] == ["absolute-path"]


def test_flags_a_home_directory_without_a_trailing_slash(repo):
    """The home rule carries a username, so it is not a SITE_ROOTS entry."""
    rel = repo("run.sh", 'DATA="/home/someone"\n')
    assert [v.rule for v in scan_content([rel])] == ["absolute-path"]


def test_every_pinned_root_is_still_declared():
    """A root may be added to the checker, never quietly dropped."""
    missing = sorted(set(GENERIC_ROOTS) - set(mod.SITE_ROOTS))
    assert not missing, f"roots no longer declared in the checker: {missing}"
    missing = sorted(set(SITE_ROOTS) - set(mod.site_roots()))
    assert not missing, f"configured roots not applied: {missing}"


def test_no_path_pattern_is_a_hand_rolled_root():
    """Every site root must be built by ``_root()``.

    A root added as a bare ``re.compile(r"/xyz/")`` behaves unlike its
    neighbours -- it misses the bare and sentence-final spellings -- and no
    behavioural test would notice, because those only know about roots that
    were declared properly. This is the check that makes the inconsistency
    impossible to reintroduce rather than merely fixed once.
    """
    built = {mod._root(name).pattern for name in mod.site_roots()}
    for pattern in mod.path_patterns():
        assert pattern.pattern in built or pattern.pattern == HOME_PATTERN, (
            f"{pattern.pattern!r} is not built by _root(): a hand-rolled "
            "root misses the bare and sentence-final spellings"
        )


@pytest.mark.parametrize("home", ["/home/Someone/x", "/home/2someone/x"])
def test_flags_home_directories_the_lowercase_class_missed(repo, home):
    rel = repo("run.sh", f"LOG={home}\n")
    assert [v.rule for v in scan_content([rel])] == ["absolute-path"]


@pytest.mark.parametrize(
    "root", ["/Users/someone/x", "/sitex/user/someone/x", "/clusterfs/005/x"]
)
def test_flags_the_other_plausible_deployment_roots(repo, root):
    rel = repo("run.sh", f"DATA={root}\n")
    assert [v.rule for v in scan_content([rel])] == ["absolute-path"]


@pytest.mark.parametrize(
    "line",
    [
        # A computing-resource acknowledgement is a near-certain addition to
        # a published research repo, and it belongs in the highest-churn file
        # in the tree. Firing on it would force either a permanent exemption
        # there or a weakened pattern under pressure.
        "Computing provided by Site X: https://sitex.example.edu/",
        "Storage documentation: https://clusterfs.example.edu/about",
        "Log in at login.sitex.example.edu with your credentials.",
        "Log in at sitex-login.example.edu with your credentials.",
        # Safe because /sitexy is not /sitex, and must stay safe.
        'docs: "https://example.org/sitexy/page"',
        # The period is a separator only when no name character follows it,
        # so these stay clean while a sentence-final path does not.
        "Mirrored to clusterfs.example.edu overnight.",
    ],
)
def test_a_hostname_is_not_a_filesystem_root(repo, line):
    rel = repo("README.md", line + "\n")
    assert scan_content([rel]) == []


@pytest.mark.parametrize("root", SITE_ROOTS)
def test_flags_every_site_root_at_the_end_of_a_sentence(repo, root):
    """A full stop must not excuse what a comma catches.

    ``.`` is both sentence punctuation and part of a hostname, so an earlier
    lookahead treated it as a name character throughout and let
    ``stored on /sitex.`` pass -- the plainest possible spelling of exactly
    the disclosure this rule exists to catch. Parametrised over this file's
    own ``SITE_ROOTS`` for the same reason as the bare-root test above.
    """
    rel = repo("README.md", f"Raw data is stored on /{root}.\n")
    assert [v.rule for v in scan_content([rel])] == ["absolute-path"]


@pytest.mark.parametrize(
    "line",
    [
        "Everything lives under /sitex. Nothing is copied.",  # mid-line stop
        "Held on /sitex, which is shared.",  # the comma always worked
    ],
)
def test_flags_a_path_followed_by_other_punctuation(repo, line):
    rel = repo("README.md", line + "\n")
    assert [v.rule for v in scan_content([rel])] == ["absolute-path"]


# --- rule class 2: acquisition identifiers ---------------------------------


def test_flags_acquisition_identifiers(repo):
    rel = repo("a.py", 'subject = "RAWID_XX"\n')
    assert [v.rule for v in scan_content([rel])] == ["identifier"]


def test_flags_hyphenated_acquisition_identifiers(repo):
    rel = repo("a.csv", "Question,RAWID-XX,RAWID-YY\n")
    assert [v.rule for v in scan_content([rel])] == ["identifier"]


def test_flags_bare_prefix_without_a_number(repo):
    """The prefix alone is enough: it is the tell that the scheme leaked."""
    rel = repo("a.py", "cols = [c for c in df if c.startswith('RAWID')]\n")
    assert [v.rule for v in scan_content([rel])] == ["identifier"]


# --- rule class 3: contact addresses ---------------------------------------


def test_flags_email_addresses(repo):
    rel = repo("run.sh", "--mail-user=someone@example.org\n")
    assert [v.rule for v in scan_content([rel])] == ["email"]


def test_published_contact_allowed_only_in_its_file(repo):
    address = next(iter(mod.PUBLISHED_CONTACTS["README.md"]))
    readme = repo("README.md", f"Corresponding author: A ({address})\n")
    assert scan_content([readme]) == []
    other = repo("notes.md", f"Contact {address}\n")
    assert [v.rule for v in scan_content([other])] == ["email"]
    mixed = repo("README.md", f"{address} or someone@example.org\n")
    assert [v.rule for v in scan_content([mixed])] == ["email"]
    meta = repo("pyproject.toml", f'{{name = "A", email = "{address}"}},\n')
    assert scan_content([meta]) == []
    assert set(mod.PUBLISHED_CONTACTS) == {"README.md", "pyproject.toml"}


# --- rule class 4: agent-directed prose ------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "Ask Claude to regenerate the figure.",
        "Generated with an Anthropic model.",
        "Copilot suggested this loop.",
        "Pasted from ChatGPT.",
        "This assertion is load-bearing.",
        "This assertion is load bearing.",
    ],
)
def test_flags_assistant_names_and_register_tells(repo, line):
    rel = repo("README.md", line + "\n")
    assert [v.rule for v in scan_content([rel])] == ["agent-prose"]


# --- the pathname rules ----------------------------------------------------


def test_flags_an_identifier_in_a_filename(repo):
    """A clean file under a leaking name is still a leak."""
    rel = repo("data/RAWID_XX_stroop_betas.csv", "subject,beta\nS1,0.4\n")
    assert scan_content([rel]) == []  # contents are spotless
    assert [v.rule for v in scan_filenames([rel])] == ["identifier-in-filename"]


def test_flags_a_deployment_root_in_a_filename(repo):
    rel = repo("mirror/home/someone/notes.md", "A clean note.\n")
    assert scan_content([rel]) == []
    assert [v.rule for v in scan_filenames([rel])] == ["path-in-filename"]


def test_a_clean_filename_scan_is_quiet(repo):
    assert scan_filenames(["scripts/step1_glm_stroop.py", "config/x.json"]) == []


def test_the_path_rules_treat_a_top_level_and_a_nested_directory_alike(repo):
    """ls-files yields relative paths, so the leading slash has to be added.

    Without it the nested spelling was flagged and the top-level one was
    not, and a rule inconsistent about depth is worse than a strict one.
    """
    assert [v.rule for v in scan_filenames(["sitex/x.py"])] == ["path-in-filename"]
    assert [v.rule for v in scan_filenames(["scripts/sitex/x.py"])] == [
        "path-in-filename"
    ]


def test_the_filename_rules_reach_files_the_content_scan_skips(repo, run_main):
    """A binary-suffix file is never opened, so only the name can catch it."""
    repo("figures/RAWID_XX_topomap.png", b"\x89PNG\r\n\x1a\n not really a png")
    code, out = run_main()
    assert code == 1
    assert "identifier-in-filename" in out
    assert "unscannable" not in out  # .png is allowlisted, so not reported twice


# --- clean content stays clean ---------------------------------------------


def test_passes_clean_content(repo):
    rel = repo(
        "ok.py",
        'SUBJECT = "S5"\nROOT = os.environ["BASE_DIR"]\n'
        '# Reads $BASE_DIR/data/{task}/NeuroLume_S5_000000a_5.snirf\n',
    )
    assert scan_content([rel]) == []


def test_allows_a_path_shape_in_docs(repo):
    """R1 permits the shape of a path, only never a real root."""
    rel = repo("README.md", "Outputs land in `$OUTPUT_DIR/03_glm/<subject>/`.\n")
    assert scan_content([rel]) == []


def test_reports_every_violation_not_just_the_first(repo):
    rel = repo("bad.sh", "A=/sitex/x\nB=someone@example.org\nC=RAWID_XX\n")
    assert sorted(v.rule for v in scan_content([rel])) == [
        "absolute-path",
        "email",
        "identifier",
    ]


# --- the exemptions, tested through behaviour ------------------------------


def test_the_checker_and_its_suite_are_not_content_scanned(repo, run_main):
    """A detector has to contain what it detects; so does its suite."""
    repo("scripts/check_public_hygiene.py", 'P = "RAWID"  # the pattern\n')
    repo("tests/test_check_public_hygiene.py", 'x = "/sitex/x"  # a fixture\n')
    assert "scripts/check_public_hygiene.py" not in tracked_text_files()
    code, out = run_main()
    assert code == 0, out


def test_the_lockfile_is_exempt_from_paths_only(repo):
    """Only absolute-path is justified there; the other three are not."""
    rel = repo("uv.lock", "path = /sitex/data/example/x\n")
    assert scan_content([rel]) == []

    rel = repo("uv.lock", "name = RAWID_XX\ncontact = someone@example.org\n")
    assert sorted(v.rule for v in scan_content([rel])) == ["email", "identifier"]


def test_gitignore_may_name_the_assistant_dir(repo):
    """Naming the assistant dir to keep it untracked is the rule working."""
    rel = repo(".gitignore", ".claude/\nCLAUDE.md\n")
    assert scan_content([rel]) == []


def test_gitignore_is_not_identifier_exempt(repo):
    rel = repo(".gitignore", "config/RAWID_map.csv\n")
    assert [v.rule for v in scan_content([rel])] == ["identifier"]


def test_the_example_map_is_not_identifier_exempt(repo):
    """It holds RAW_01,S1 and no identifier, so it needs no exemption."""
    rel = repo("config/subject_map.csv.example", "raw_id,code\nRAWID_XX,S1\n")
    assert [v.rule for v in scan_content([rel])] == ["identifier"]


def test_an_allowlisted_binary_suffix_is_skipped_not_reported(repo, run_main):
    repo("data/betas.npz", b"\x93NUMPY\x01\x00 \xff\xfe not utf-8")
    code, out = run_main()
    assert code == 0, out


@pytest.mark.parametrize("name", ["betas.npy", "rois.parquet", "raw.fif"])
def test_the_formats_this_project_actually_writes_are_allowlisted(
    repo, run_main, name
):
    """.npz was allowlisted and .npy was not; .fif is MNE-native."""
    repo(f"data/{name}", b"\xff\xfe\x00 binary payload")
    code, out = run_main()
    assert code == 0, out


# --- the three non-pattern assertions --------------------------------------


def test_reports_an_unscannable_tracked_file(repo):
    """Silence is the worst answer a gate can give."""
    rel = repo("roster.xlsx", b"PK\x03\x04\xff\xfe binary, holds who knows what")
    found = scan_content([rel])
    assert [v.rule for v in found] == ["unscannable"]
    assert "BINARY_SUFFIXES" in found[0].text


def test_an_unscannable_file_fails_the_whole_run(repo, run_main):
    repo("roster.xlsx", b"\xff\xfe\x00 not text")
    code, out = run_main()
    assert code == 1
    assert "roster.xlsx:0: unscannable" in out


def test_main_reports_a_tracked_file_under_docs(repo, run_main):
    repo("docs/manuscript/results.md", "Entirely clean prose.\n")
    code, out = run_main()
    assert code == 1
    assert "unpublished-doc" in out


def test_the_tracked_floor_rejects_an_implausibly_small_tree():
    found = check_tracked_floor(["README.md"])
    assert [v.rule for v in found] == ["too-few-tracked-files"]
    assert "not the repository tree" in found[0].text


def test_the_tracked_floor_accepts_a_full_tree():
    assert check_tracked_floor([f"f{n}" for n in range(mod.MIN_TRACKED_FILES)]) == []


def test_an_empty_index_does_not_pass_vacuously(repo, run_main):
    """git init with nothing tracked scanned clean and exited 0."""
    code, out = run_main(pad=False)
    assert code == 1
    assert "too-few-tracked-files" in out


# --- the invariant the whole task exists to hold ---------------------------


def test_the_real_tree_holds_no_configured_site_value():
    """No configured root or identifier anywhere, exempt files included.

    The checker and this file are exempt from the content rules, which makes
    them the two places a site value could land unnoticed. The values come
    from the real settings (environment, then the repository ``.env``), so
    the test is skipped where none are configured.
    """
    root = Path(__file__).resolve().parents[1]
    env = {**mod.read_local_settings(root), **_REAL_SETTINGS}
    if not mod.settings_configured(env):
        pytest.skip("HYGIENE_SITE_ROOTS and HYGIENE_ID_PATTERNS not set")
    patterns = [
        *mod.identifier_patterns(env),
        *(mod._root(r) for r in mod.site_roots(env) if r not in mod.SITE_ROOTS),
    ]
    hits = []
    # Tracked files when this is a git checkout; every file in an exported
    # copy, which has no .git directory.
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files"], capture_output=True, text=True
    )
    if listed.returncode == 0:
        paths = [root / rel for rel in listed.stdout.splitlines()]
    else:
        paths = [p for p in root.rglob("*") if "__pycache__" not in p.parts]
    for path in paths:
        if not path.is_file() or path.name == ".env":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits += [
            f"{path.relative_to(root)}:{n}"
            for n, line in enumerate(text.splitlines(), 1)
            if any(pattern.search(line) for pattern in patterns)
        ]
    assert hits == [], f"configured site values present: {hits}"


# --- the local settings ------------------------------------------------------


def test_settings_are_comma_lists_with_blanks_ignored(monkeypatch):
    monkeypatch.setenv("HYGIENE_SITE_ROOTS", " sitex, ,/clusterfs/,,sitex")
    monkeypatch.setenv("HYGIENE_ID_PATTERNS", "RAWID; ;OTHERID[-_]?[0-9]+;")
    assert mod.site_roots() == ("Users", "sitex", "clusterfs")
    patterns = mod.identifier_patterns()
    assert [p.pattern for p in patterns] == ["RAWID", "OTHERID[-_]?[0-9]+"]
    assert all(p.flags & re.IGNORECASE for p in patterns)
    assert mod.identifier_patterns()[0].search("rawid_xx")


def test_identifier_patterns_split_on_semicolons_not_commas(repo, monkeypatch):
    """A comma inside a regex quantifier must not split the pattern."""
    monkeypatch.setenv("HYGIENE_ID_PATTERNS", "RAWID[0-9]{2,3};OTHERID")
    assert [p.pattern for p in mod.identifier_patterns()] == [
        "RAWID[0-9]{2,3}",
        "OTHERID",
    ]
    rel = repo("a.txt", "x RAWID12\ny RAWID1\nz otherid\n")
    assert [(v.line, v.rule) for v in scan_content([rel])] == [
        (1, "identifier"),
        (3, "identifier"),
    ]


def test_configured_roots_are_built_like_the_generic_ones(monkeypatch):
    monkeypatch.setenv("HYGIENE_SITE_ROOTS", "sitex")
    assert mod._root("sitex") in mod.path_patterns()
    assert mod.path_patterns()[-1].pattern == HOME_PATTERN


@pytest.mark.parametrize(
    "roots, patterns", [("", ""), (" ", " "), (" , ,", " ; ;")]
)
def test_blank_settings_count_as_unset(monkeypatch, roots, patterns):
    monkeypatch.setenv("HYGIENE_SITE_ROOTS", roots)
    monkeypatch.setenv("HYGIENE_ID_PATTERNS", patterns)
    assert not mod.settings_configured()
    assert mod.site_roots() == GENERIC_ROOTS
    assert mod.identifier_patterns() == ()


def test_unset_settings_give_generic_defaults(repo, unset_settings):
    assert not mod.settings_configured()
    assert mod.site_roots() == GENERIC_ROOTS
    assert mod.identifier_patterns() == ()
    rel = repo(
        "run.sh",
        "A=/sitex/x\nB=RAWID_XX\nC=/Users/someone/x\nD=/home/someone/x\n",
    )
    assert [(v.line, v.rule) for v in scan_content([rel])] == [
        (3, "absolute-path"),
        (4, "absolute-path"),
    ]


def test_main_reports_generic_defaults_when_unset(run_main, unset_settings):
    code, out = run_main()
    assert code == 0
    assert mod.GENERIC_DEFAULTS_MESSAGE in out


def test_main_is_quiet_about_defaults_when_configured(run_main):
    code, out = run_main()
    assert code == 0
    assert mod.GENERIC_DEFAULTS_MESSAGE not in out


def test_main_reads_the_settings_from_the_repository_env_file(
    repo, run_main, unset_settings, tmp_path
):
    (tmp_path / ".env").write_text(
        "HYGIENE_SITE_ROOTS=sitex\nHYGIENE_ID_PATTERNS=RAWID\n"
    )
    repo("leak.sh", "A=/sitex/x\nB=RAWID_XX\n")
    code, out = run_main()
    assert code == 1
    assert "leak.sh:1: absolute-path" in out
    assert "leak.sh:2: identifier" in out
    assert mod.GENERIC_DEFAULTS_MESSAGE not in out


def test_the_environment_takes_precedence_over_the_env_file(
    repo, run_main, monkeypatch, tmp_path
):
    """``load_dotenv(override=False)``: a variable already set wins."""
    monkeypatch.setenv("HYGIENE_SITE_ROOTS", "clusterfs")
    (tmp_path / ".env").write_text("HYGIENE_SITE_ROOTS=sitex\n")
    repo("ok.sh", "A=/sitex/x\n")
    code, out = run_main()
    assert code == 0, out


def test_a_blank_environment_value_does_not_hide_the_env_file(
    repo, run_main, monkeypatch, tmp_path
):
    """``export HYGIENE_SITE_ROOTS=`` counts as unset, so ``.env`` still applies."""
    for name in SETTING_NAMES:
        monkeypatch.setenv(name, "")
    (tmp_path / ".env").write_text(
        "HYGIENE_SITE_ROOTS=sitex\nHYGIENE_ID_PATTERNS=RAWID\n"
    )
    repo("leak.sh", "A=/sitex/x\nB=RAWID_XX\n")
    code, out = run_main()
    assert code == 1
    assert "leak.sh:1: absolute-path" in out
    assert "leak.sh:2: identifier" in out


@pytest.mark.parametrize(
    "present, missing",
    [
        ((), ["HYGIENE_SITE_ROOTS", "HYGIENE_ID_PATTERNS"]),
        (("HYGIENE_SITE_ROOTS",), ["HYGIENE_ID_PATTERNS"]),
        (("HYGIENE_ID_PATTERNS",), ["HYGIENE_SITE_ROOTS"]),
    ],
)
def test_require_settings_fails_naming_what_is_missing(
    repo, monkeypatch, capsys, unset_settings, present, missing
):
    values = {"HYGIENE_SITE_ROOTS": "sitex", "HYGIENE_ID_PATTERNS": "RAWID"}
    for name in present:
        monkeypatch.setenv(name, values[name])
    for n in range(mod.MIN_TRACKED_FILES):
        repo(f"filler/f{n}.txt", f"clean line {n}\n")
    monkeypatch.setattr(sys, "argv", ["check_public_hygiene.py", "--require-settings"])
    code = mod.main()
    err = capsys.readouterr().err
    assert code != 0
    for name in missing:
        assert name in err
    for name in present:
        assert name not in err


def test_require_settings_passes_when_both_are_set(repo, run_main, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["check_public_hygiene.py", "--require-settings"]
    )
    for n in range(mod.MIN_TRACKED_FILES):
        repo(f"filler/f{n}.txt", f"clean line {n}\n")
    monkeypatch.delenv("BASE_DIR", raising=False)
    monkeypatch.delenv("OUD_TPBM_BASE_DIR", raising=False)
    assert mod.main() == 0


def test_an_invalid_identifier_pattern_exits_with_one_line(
    repo, monkeypatch, capsys
):
    monkeypatch.setenv("HYGIENE_ID_PATTERNS", "RAWID[")
    monkeypatch.setattr(sys, "argv", ["check_public_hygiene.py"])
    code = mod.main()
    captured = capsys.readouterr()
    assert code == 2
    assert "HYGIENE_ID_PATTERNS" in captured.err
    assert len(captured.err.strip().splitlines()) == 1


def test_violation_renders_as_a_locatable_line():
    v = Violation("a.py", 7, "identifier", 'x = "RAWID_XX"')
    assert str(v).startswith("a.py:7: identifier:")


# --- rule class 5: real session IDs, read at run time ------------------------

SYNTHETIC_ID = "abcdef1"
EXEMPT_PATHS = ("scripts/check_public_hygiene.py", "tests/test_check_public_hygiene.py")


@pytest.fixture
def study_data(tmp_path, monkeypatch):
    """A synthetic study root whose task_info CSVs list one session ID."""
    root = tmp_path / "study"
    (root / "data").mkdir(parents=True)
    for task in ("stroop", "wcst"):
        (root / "data" / f"{task}_task_info.csv").write_text(
            f"filename,subj_id,task_id,task_order\nf.snirf,S1,{SYNTHETIC_ID},1\n"
        )
    monkeypatch.setenv("OUD_TPBM_BASE_DIR", str(root))
    monkeypatch.delenv("BASE_DIR", raising=False)
    return root


def test_session_ids_load_from_the_task_info_csvs(repo, study_data):
    assert mod.load_session_ids() == frozenset({SYNTHETIC_ID})


def test_session_ids_fall_back_to_base_dir(repo, study_data, monkeypatch):
    monkeypatch.setenv("BASE_DIR", str(study_data))
    monkeypatch.delenv("OUD_TPBM_BASE_DIR")
    assert mod.load_session_ids() == frozenset({SYNTHETIC_ID})


def test_session_ids_unavailable_without_study_data(repo, monkeypatch):
    monkeypatch.delenv("OUD_TPBM_BASE_DIR", raising=False)
    monkeypatch.delenv("BASE_DIR", raising=False)
    assert mod.load_session_ids() is None


def test_session_ids_fail_closed_when_one_csv_is_missing(repo, study_data):
    (study_data / "data" / "wcst_task_info.csv").unlink()
    with pytest.raises(mod.SessionIdSourceError, match="wcst_task_info.csv is missing"):
        mod.load_session_ids()


def test_session_ids_fail_closed_without_the_id_column(repo, study_data):
    (study_data / "data" / "stroop_task_info.csv").write_text("filename,subj_id\nf.snirf,S1\n")
    with pytest.raises(mod.SessionIdSourceError, match="no task_id column"):
        mod.load_session_ids()


def test_session_ids_fail_closed_on_an_empty_file(repo, study_data):
    (study_data / "data" / "wcst_task_info.csv").write_text("")
    with pytest.raises(mod.SessionIdSourceError, match="wcst_task_info.csv"):
        mod.load_session_ids()


def test_session_ids_fail_closed_on_a_header_only_file(repo, study_data):
    (study_data / "data" / "wcst_task_info.csv").write_text("filename,subj_id,task_id\n")
    with pytest.raises(mod.SessionIdSourceError, match="yields no session IDs"):
        mod.load_session_ids()


def test_session_ids_fail_closed_on_undecodable_bytes(repo, study_data):
    (study_data / "data" / "stroop_task_info.csv").write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(mod.SessionIdSourceError, match="cannot be read"):
        mod.load_session_ids()


def test_session_ids_fail_closed_on_an_unreadable_file(repo, study_data):
    import os

    path = study_data / "data" / "stroop_task_info.csv"
    path.chmod(0)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("permissions are not enforced for this user (root)")
        with pytest.raises(mod.SessionIdSourceError, match="cannot be read"):
            mod.load_session_ids()
    finally:
        path.chmod(0o644)


def test_a_broken_first_root_is_not_replaced_by_a_later_one(repo, study_data, tmp_path, monkeypatch):
    good = tmp_path / "good"
    (good / "data").mkdir(parents=True)
    for task in ("stroop", "wcst"):
        (good / "data" / f"{task}_task_info.csv").write_text(
            "filename,subj_id,task_id,task_order\nf.snirf,S1,1234567,1\n"
        )
    (study_data / "data" / "stroop_task_info.csv").write_text("filename,subj_id\n")
    monkeypatch.setenv("BASE_DIR", str(good))  # OUD_TPBM_BASE_DIR (broken) comes first
    with pytest.raises(mod.SessionIdSourceError):
        mod.load_session_ids()


def test_a_root_without_task_info_files_is_skipped_over(repo, study_data, tmp_path, monkeypatch):
    monkeypatch.setenv("OUD_TPBM_BASE_DIR", str(tmp_path / "nothing_here"))
    monkeypatch.setenv("BASE_DIR", str(study_data))
    assert mod.load_session_ids() == frozenset({SYNTHETIC_ID})


def test_main_exits_nonzero_with_a_message_not_a_traceback_on_a_broken_source(
    repo, study_data, monkeypatch, capsys
):
    (study_data / "data" / "wcst_task_info.csv").write_text("filename,subj_id\n")
    for n in range(mod.MIN_TRACKED_FILES):
        repo(f"filler/f{n}.txt", f"clean line {n}\n")
    monkeypatch.setattr(sys, "argv", ["check_public_hygiene.py"])
    code = mod.main()
    out = capsys.readouterr().out
    assert code == 1
    assert "session-id-source" in out and "no task_id column" in out
    assert mod.SESSION_ID_SKIP_MESSAGE not in out


def test_other_rules_do_not_echo_a_real_session_id(repo, study_data, monkeypatch, capsys):
    for n in range(mod.MIN_TRACKED_FILES):
        repo(f"filler/f{n}.txt", f"clean line {n}\n")
    repo("x.sh", f"P=/sitex/data/S1_{SYNTHETIC_ID}_glm_contrasts.csv\n")
    monkeypatch.setattr(sys, "argv", ["check_public_hygiene.py"])
    code = mod.main()
    out = capsys.readouterr().out
    assert code == 1 and "absolute-path" in out
    assert SYNTHETIC_ID not in out


def test_flags_a_real_session_id_in_content_and_redacts_it(repo):
    rel = repo("notes.txt", f"file S1_{SYNTHETIC_ID}_5.snirf\nother 1234567 line\n")
    found = mod.scan_session_ids([rel], frozenset({SYNTHETIC_ID}))
    assert [(v.rule, v.line) for v in found] == [("session-id", 1)]
    assert SYNTHETIC_ID not in str(found[0])


def test_flags_a_real_session_id_in_a_filename(repo):
    rel = repo(f"out/S1_{SYNTHETIC_ID}_results.json", "{}\n")
    found = mod.scan_session_ids([rel], frozenset({SYNTHETIC_ID}))
    assert {v.rule for v in found} == {"session-id-in-filename"}
    assert SYNTHETIC_ID not in str(found[0])


def test_session_id_rule_has_no_exemption_for_the_checker_or_its_suite(repo):
    rels = [repo(p, f"{SYNTHETIC_ID}\n") for p in EXEMPT_PATHS]
    found = mod.scan_session_ids(rels, frozenset({SYNTHETIC_ID}))
    assert sorted(v.path for v in found) == sorted(rels)


def test_session_id_match_respects_token_boundaries(repo):
    rel = repo("x.txt", f"a{SYNTHETIC_ID} {SYNTHETIC_ID}0 deadbee\n")
    assert mod.scan_session_ids([rel], frozenset({SYNTHETIC_ID})) == []


def test_main_fails_on_a_session_id_when_study_data_is_present(repo, study_data, monkeypatch, capsys):
    for n in range(mod.MIN_TRACKED_FILES):
        repo(f"filler/f{n}.txt", f"clean line {n}\n")
    repo("leak.txt", f"{SYNTHETIC_ID}\n")
    monkeypatch.setattr(sys, "argv", ["check_public_hygiene.py"])
    code = mod.main()
    out = capsys.readouterr().out
    assert code == 1 and "session-id" in out and SYNTHETIC_ID not in out
    assert mod.SESSION_ID_SKIP_MESSAGE not in out


def test_main_reports_the_skip_without_study_data(run_main):
    code, out = run_main()
    assert code == 0
    assert "session-ID rule skipped: study data not available" in out


def test_outside_a_git_checkout_exits_with_one_line(tmp_path, monkeypatch, capsys):
    """Run outside a git checkout, the checker says so instead of a traceback."""
    outside = tmp_path / "not_a_repo"
    outside.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.setattr(mod, "REPO_ROOT", outside)
    monkeypatch.setattr(sys, "argv", ["check_public_hygiene.py"])
    code = mod.main()
    captured = capsys.readouterr()
    assert code != 0
    assert "must run inside" in captured.err
    assert len(captured.err.strip().splitlines()) == 1
    assert "Traceback" not in captured.err + captured.out
