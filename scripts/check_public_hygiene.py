#!/usr/bin/env python3
"""Check tracked files against the public-repository content rules.

This repository is published. Five classes of content are easy to add
without noticing and hard to remove once pushed, because history is
append-only:

1. **Absolute paths.** They disclose the account, the cluster allocation
   and where the study data live. Paths belong in the
   gitignored ``.env`` and are read from the environment. The *shape* of a
   path is fine; a real root is not. ``/Users`` and ``/home/<user>`` are
   checked everywhere; a deployment's own roots come from
   ``HYGIENE_SITE_ROOTS``.
2. **Acquisition identifiers.** Subjects are ``S1`` ... ``S12`` in every
   tracked file. The raw identifier scheme must not appear anywhere; its
   patterns come from ``HYGIENE_ID_PATTERNS``.
3. **Contact addresses.** An email in a tracked file is a durable
   publication of it.
4. **Agent-directed prose.** Documentation here addresses someone
   reproducing the work; text written to or about an assistant tells that
   reader the docs were not written for them.
5. **Acquisition session IDs.** The 7-hex session IDs listed in the study's
   ``task_info`` CSVs. The checker never stores them, or hashes of them: it
   reads them at run time from ``data/{stroop,wcst}_task_info.csv`` under
   ``OUD_TPBM_BASE_DIR`` or ``BASE_DIR`` (environment, or ``BASE_DIR`` in the
   repository's ``.env``). With no study data found the rule is skipped and
   the run says so; a skip is not a violation, so a run without the data
   does not check this class. If study data is found but unusable (a missing
   or unreadable file, no ``task_id`` column, no IDs) the run fails.

``HYGIENE_SITE_ROOTS`` and ``HYGIENE_ID_PATTERNS`` are read from the environment or
the untracked ``.env``, so the values that describe one site are never tracked.
With neither set, the run says the two rules use generic defaults only.

Classes 1 and 2 are checked against each tracked *pathname* as well as its
contents, and so is class 5: a clean file under a leaking name is still a
leak.

Three further assertions, each closing a way for the check to pass without
having checked anything:

- **No tracked file under ``docs/``.** The manuscript lives there and is
  kept out of the release, so a tracked file under that path is the signal
  that a blanket ``git add`` pulled it in.
- **No unscannable tracked file.** A file that does not decode as UTF-8 and
  is not in ``BINARY_SUFFIXES`` is reported rather than skipped. A tracked
  ``roster.xlsx`` holding an identifier would otherwise pass in silence.
  Allowlist a suffix deliberately; do not let the checker drop it quietly.

  **``BINARY_SUFFIXES`` is a trust boundary, not a proof.** An allowlisted
  binary with a clean filename is never inspected at all, and neither
  ``git grep -I`` nor this checker can see inside it: a ``.csv.gz`` of
  per-subject betas, a ``.npz`` of state betas or a ``.pdf`` can carry
  identifiers straight past the gate. ``.gz``, ``.pkl``, ``.npz`` and
  ``.pdf`` are the live exposures. No regex closes this; check such files
  by hand or with a format-aware reader.
- **A floor on the tracked-file count.** An empty or near-empty index
  scans clean, so a checker pointed at the wrong directory would report
  success. Below ``MIN_TRACKED_FILES`` the run fails instead.

Run directly::

    python scripts/check_public_hygiene.py

Exits 0 when clean and 1 on any violation, printing one line per violation
as ``path:line: rule: text``. Pathname and whole-file findings report line 0.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Mapping, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

def _root(name: str) -> "re.Pattern[str]":
    """Match a named filesystem root, but not a hostname starting with it.

    A root needs no trailing slash to disclose the site: ``BASE="/data"`` and
    ``DATA="/home/someone"`` are ordinary ways to write one in a shell
    variable. A bare ``/cluster`` would also match
    ``https://cluster.example.edu``, so the next character must be a
    separator rather than a name character. That admits ``/cluster/data/x``
    and ``"/cluster"`` and rejects ``cluster.example.edu`` and
    ``cluster-login.example.edu``, so a citation of the host in an
    acknowledgement passes.

    A period is both sentence punctuation and part of a hostname. It counts
    as a separator only when no name character follows it: ``Raw data is
    stored on /cluster.`` is a disclosure, while ``cluster.example.edu`` is a
    hostname. A comma and a full stop are treated alike. Both halves of the
    period rule are pinned by tests.
    """
    return re.compile(
        rf"/{re.escape(name)}(?=/|\Z|\.(?![A-Za-z0-9])|[^A-Za-z0-9._-])"
    )


# Settings that name a deployment's own roots and identifier scheme. Those
# values describe one site, so they live in the untracked ``.env`` (or the
# environment), never in this file. Roots are separated by commas; patterns
# by semicolons, because a regex may hold a comma (``ID[0-9]{2,3}``). Blank
# entries are ignored, and a variable set to a blank value counts as unset.
#
#   HYGIENE_SITE_ROOTS=data,cluster          directory names, each built by _root()
#   HYGIENE_ID_PATTERNS=RAWID[-_]?[0-9]+;X1  regexes, matched case-insensitively
SITE_ROOTS_ENV = "HYGIENE_SITE_ROOTS"
ID_PATTERNS_ENV = "HYGIENE_ID_PATTERNS"
SETTING_NAMES = (SITE_ROOTS_ENV, ID_PATTERNS_ENV)
_SEPARATORS = {SITE_ROOTS_ENV: ",", ID_PATTERNS_ENV: ";"}
GENERIC_DEFAULTS_MESSAGE = (
    "site-root and identifier rules: generic defaults only "
    f"({SITE_ROOTS_ENV} and {ID_PATTERNS_ENV} not set)"
)

# Roots checked everywhere, whatever the local settings say. Every root,
# these and the configured ones alike, goes through _root(), so a bare
# `X="/data"` and a sentence-final `runs on /data.` are caught the same way.
# Add a root by name, never as a hand-rolled pattern.
SITE_ROOTS = ("Users",)

# Any username, not just a lowercase-leading one: /home/Someone/ and
# /home/2someone/ disclose an account just as well.
HOME_PATTERN = re.compile(r"/home/[A-Za-z0-9][A-Za-z0-9_-]*")

# Identifier patterns checked everywhere. None is generic, so the scheme to
# reject comes from HYGIENE_ID_PATTERNS. A prefix alone is a valid pattern:
# code that tests for it names it as surely as a literal identifier does.
IDENTIFIER_PATTERNS: tuple["re.Pattern[str]", ...] = ()


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal ``KEY=VALUE`` reader, used only when python-dotenv is absent."""
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip()] = value.strip().strip("\"'")
    return values


def read_local_settings(root: Path | None = None) -> dict[str, str]:
    """Return the repository ``.env`` as a mapping, without touching the environment."""
    env_file = (root or REPO_ROOT) / ".env"
    if not env_file.is_file():
        return {}
    try:
        from dotenv import dotenv_values
    except ImportError:
        return _parse_env_file(env_file)
    return {k: v for k, v in dotenv_values(env_file).items() if v is not None}


def load_local_settings(root: Path | None = None) -> None:
    """Load the repository ``.env``; variables already set take precedence.

    A ``HYGIENE_*`` variable set to a blank value is removed first, so an
    empty export in the shell does not hide the value in ``.env``.
    """
    for name in SETTING_NAMES:
        if name in os.environ and not os.environ[name].strip():
            del os.environ[name]
    env_file = (root or REPO_ROOT) / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        for key, value in read_local_settings(root).items():
            os.environ.setdefault(key, value)
        return
    if env_file.is_file():
        load_dotenv(env_file, override=False)


def _setting_list(name: str, env: "Mapping[str, str] | None" = None) -> list[str]:
    raw = (os.environ if env is None else env).get(name) or ""
    return [item.strip() for item in raw.split(_SEPARATORS[name]) if item.strip()]


def missing_settings(env: "Mapping[str, str] | None" = None) -> list[str]:
    """The ``HYGIENE_*`` settings that name no entry."""
    return [name for name in SETTING_NAMES if not _setting_list(name, env)]


def site_roots(env: "Mapping[str, str] | None" = None) -> tuple[str, ...]:
    """The generic roots plus those named in ``HYGIENE_SITE_ROOTS``."""
    configured = (item.strip("/") for item in _setting_list(SITE_ROOTS_ENV, env))
    extra = tuple(dict.fromkeys(r for r in configured if r and r not in SITE_ROOTS))
    return SITE_ROOTS + extra


def path_patterns(env: "Mapping[str, str] | None" = None) -> tuple["re.Pattern[str]", ...]:
    """Every absolute-path pattern: one per site root, plus ``/home/<user>``."""
    return (*(_root(name) for name in site_roots(env)), HOME_PATTERN)


def identifier_patterns(
    env: "Mapping[str, str] | None" = None,
) -> tuple["re.Pattern[str]", ...]:
    """The generic patterns plus those in ``HYGIENE_ID_PATTERNS``, case-insensitive."""
    return IDENTIFIER_PATTERNS + tuple(
        re.compile(p, re.IGNORECASE) for p in _setting_list(ID_PATTERNS_ENV, env)
    )


def settings_configured(env: "Mapping[str, str] | None" = None) -> bool:
    """True when either local setting names at least one entry."""
    return bool(_setting_list(SITE_ROOTS_ENV, env) or _setting_list(ID_PATTERNS_ENV, env))


EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Addresses published on purpose, per file. The corresponding author's
# contact is part of the paper's front matter: the README's data statement
# points to it and the package metadata lists it. Any other address, or
# this one in another file, is still a violation.
_CORRESPONDING_AUTHOR = frozenset({"aestudillo@hopehouse.org"})
PUBLISHED_CONTACTS = {
    "README.md": _CORRESPONDING_AUTHOR,
    "pyproject.toml": _CORRESPONDING_AUTHOR,
}

# Assistant names and register tells. Kept short on purpose: a long list
# produces false positives on ordinary technical prose, and the rule is
# about who the documentation addresses, not about banning vocabulary.
PROSE_PATTERNS = (
    re.compile(r"\bclaude\b", re.IGNORECASE),
    re.compile(r"\banthropic\b", re.IGNORECASE),
    re.compile(r"\bcopilot\b", re.IGNORECASE),
    re.compile(r"\bchatgpt\b", re.IGNORECASE),
    re.compile(r"\bload[- ]bearing\b", re.IGNORECASE),
)

# Exempt from every content rule. Only these two qualify: a detector has to
# contain what it detects, and its suite has to contain a fixture of each
# banned class in order to assert the rule fires. Neither exemption covers
# the pathname rules, which apply to every tracked path without exception.
EXEMPT = frozenset(
    {
        "scripts/check_public_hygiene.py",
        "tests/test_check_public_hygiene.py",
    }
)

# Per-rule exemptions, each narrowed to the one rule it needs.
#
# .gitignore has to name the assistant directory in order to keep it
# untracked, which is the rule working rather than failing.
# A lockfile is generated and may resolve a local editable install to an
# absolute path; it has no business carrying any of the other three classes.
PROSE_EXEMPT = frozenset({".gitignore"})
PATH_EXEMPT = frozenset({"uv.lock"})

# Suffixes whose contents this checker cannot read and does not try to. Any
# other undecodable tracked file is reported, not skipped.
BINARY_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".pdf",
        ".gz",
        ".nii",
        ".snirf",
        ".pkl",
        ".npy",
        ".npz",
        ".parquet",
        ".fif",  # MNE-native, plausible once the brain pipeline ships outputs
    }
)

# A tree this small is not this repository, which tracks well over a hundred
# files; an index below this floor means the checker is not scanning what it thinks.
MIN_TRACKED_FILES = 20


class Violation(NamedTuple):
    """One rule breach, located precisely enough to fix without searching."""

    path: str
    line: int
    rule: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.text.strip()}"


def _git(*args: str) -> list[str]:
    """Run git in the repository and return its stdout as lines."""
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.split("\n") if line]


def tracked_files() -> list[str]:
    """Every tracked path, unfiltered: the input to the pathname rules."""
    return _git("ls-files")


def tracked_text_files() -> list[str]:
    """Return tracked, non-exempt paths that are plausibly text."""
    return [
        p
        for p in tracked_files()
        if p not in EXEMPT and Path(p).suffix.lower() not in BINARY_SUFFIXES
    ]


def scan_filenames(paths: list[str]) -> list[Violation]:
    """Apply the identifier and path rules to the pathnames themselves.

    A file whose contents are spotless still leaks if its name carries a
    participant identifier, so ``git ls-files`` output is checked as well as
    opened. No exemptions apply here: there is no
    legitimate reason for a tracked pathname to contain either class.

    The path rules match against ``"/" + rel``, because ``git ls-files``
    yields relative paths: without the leading slash ``scripts/data/x.py``
    would be flagged and the top-level ``data/x.py`` would not, and a rule
    that is inconsistent about depth is worse than one that is strict.
    """
    id_patterns = identifier_patterns()
    path_rules = path_patterns()
    found: list[Violation] = []
    for rel in paths:
        for pattern in id_patterns:
            if pattern.search(rel):
                found.append(Violation(rel, 0, "identifier-in-filename", rel))
                break
        for pattern in path_rules:
            if pattern.search("/" + rel):
                found.append(Violation(rel, 0, "path-in-filename", rel))
                break
    return found


def scan_content(paths: list[str]) -> list[Violation]:
    """Apply every pattern set, line by line, over ``paths``."""
    id_patterns = identifier_patterns()
    path_rules = path_patterns()
    found: list[Violation] = []
    for rel in paths:
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError) as exc:
            # Never drop a tracked file in silence. The caller already
            # removed every allowlisted binary suffix, so whatever lands
            # here is a file nobody decided not to read.
            found.append(
                Violation(
                    rel,
                    0,
                    "unscannable",
                    f"{type(exc).__name__}: not scanned; allowlist the suffix "
                    f"in BINARY_SUFFIXES or remove the file",
                )
            )
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if rel not in PATH_EXEMPT:
                for pattern in path_rules:
                    if pattern.search(line):
                        found.append(Violation(rel, lineno, "absolute-path", line))
                        break
            for pattern in id_patterns:
                if pattern.search(line):
                    found.append(Violation(rel, lineno, "identifier", line))
                    break
            allowed = PUBLISHED_CONTACTS.get(rel, frozenset())
            if any(m.group(0) not in allowed for m in EMAIL_PATTERN.finditer(line)):
                found.append(Violation(rel, lineno, "email", line))
            if rel not in PROSE_EXEMPT:
                for pattern in PROSE_PATTERNS:
                    if pattern.search(line):
                        found.append(Violation(rel, lineno, "agent-prose", line))
                        break
    return found


SESSION_ID_RE = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{7}(?![0-9A-Za-z])", re.IGNORECASE)
SESSION_ID_SKIP_MESSAGE = "session-ID rule skipped: study data not available"


def _study_base_dirs() -> list[Path]:
    """Candidate study-data roots: OUD_TPBM_BASE_DIR, BASE_DIR, then .env's BASE_DIR."""
    values = [os.environ.get("OUD_TPBM_BASE_DIR"), os.environ.get("BASE_DIR")]
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == "BASE_DIR":
                values.append(value.strip().strip("\"'"))
    return [Path(v) for v in values if v]


class SessionIdSourceError(Exception):
    """The session-ID source exists but cannot be used to check anything."""


def _read_task_ids(path: Path) -> set[str]:
    """Return the session IDs in ``path``'s ``task_id`` column, or raise."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "task_id" not in reader.fieldnames:
                raise SessionIdSourceError(f"{path.name} has no task_id column")
            ids = {
                (row.get("task_id") or "").strip().lower() for row in reader
            }
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise SessionIdSourceError(
            f"{path.name} cannot be read ({type(exc).__name__})"
        ) from None
    ids = {i for i in ids if SESSION_ID_RE.fullmatch(i)}
    if not ids:
        raise SessionIdSourceError(f"{path.name} yields no session IDs")
    return ids


def load_session_ids() -> frozenset[str] | None:
    """Return the real session IDs, or None when no study data is found.

    The IDs are read from the ``task_id`` column of
    ``data/{stroop,wcst}_task_info.csv`` and held only in memory. The first
    candidate root that has either file is the one used. From then on the rule
    fails closed: a missing second file, an unreadable or undecodable file, a
    missing ``task_id`` column, or a file with no IDs raises
    ``SessionIdSourceError`` instead of weakening the check, and no other
    candidate root is tried.
    """
    for base in _study_base_dirs():
        paths = [base / "data" / f"{task}_task_info.csv" for task in ("stroop", "wcst")]
        if not any(p.exists() for p in paths):
            continue
        ids: set[str] = set()
        for path in paths:
            if not path.is_file():
                raise SessionIdSourceError(f"{path.name} is missing under {base / 'data'}")
            ids |= _read_task_ids(path)
        return frozenset(ids)
    return None


def redact_session_ids(text: str, ids: frozenset[str]) -> str:
    """Replace every real session ID in ``text`` with a placeholder."""
    return SESSION_ID_RE.sub(
        lambda m: "<session-id>" if m.group(0).lower() in ids else m.group(0), text
    )


def scan_session_ids(paths: list[str], ids: frozenset[str]) -> list[Violation]:
    """Flag every tracked pathname and line containing a real session ID.

    No exemptions: not even this file or its test may contain one. The
    reported path and text are redacted, so the output does not reproduce the ID.
    """

    def redact(text: str) -> str:
        return redact_session_ids(text, ids)

    def has_id(text: str) -> bool:
        return any(m.group(0).lower() in ids for m in SESSION_ID_RE.finditer(text))

    found: list[Violation] = []
    for rel in paths:
        if has_id(rel):
            found.append(Violation(redact(rel), 0, "session-id-in-filename", redact(rel)))
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # undecodable files are reported by scan_content
        for lineno, line in enumerate(text.splitlines(), start=1):
            if has_id(line):
                found.append(Violation(redact(rel), lineno, "session-id", redact(line)))
    return found


def scan_docs() -> list[Violation]:
    """Assert that no file under ``docs/`` is tracked.

    The manuscript lives there and is kept out of the release. In this tree
    the exclusion holds by absence rather than by ignore rule, so this
    assertion is what detects a blanket ``git add`` having pulled the
    manuscript in.
    """
    return [
        Violation(p, 0, "unpublished-doc", "tracked under docs/")
        for p in _git("ls-files", "docs/")
    ]


def check_tracked_floor(paths: list[str]) -> list[Violation]:
    """Fail when there are too few tracked files to be the real tree.

    An empty index scans clean and exits 0, so without this a checker run
    against the wrong directory reports success.
    """
    if len(paths) >= MIN_TRACKED_FILES:
        return []
    return [
        Violation(
            "(index)",
            0,
            "too-few-tracked-files",
            f"{len(paths)} tracked files scanned, expected at least "
            f"{MIN_TRACKED_FILES}: this is not the repository tree",
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet", action="store_true", help="Print only the summary line"
    )
    parser.add_argument(
        "--require-settings",
        action="store_true",
        help=f"Fail unless {SITE_ROOTS_ENV} and {ID_PATTERNS_ENV} are both set "
        "(use for the pre-publication run)",
    )
    args = parser.parse_args()

    load_local_settings()
    try:
        identifier_patterns()
    except re.error as exc:
        print(
            f"publication hygiene: {ID_PATTERNS_ENV} holds an invalid regex ({exc})",
            file=sys.stderr,
        )
        return 2
    missing = missing_settings()
    if args.require_settings and missing:
        print(
            f"publication hygiene: --require-settings: {', '.join(missing)} not set",
            file=sys.stderr,
        )
        return 2
    if not settings_configured():
        print(GENERIC_DEFAULTS_MESSAGE)

    try:
        paths = tracked_text_files()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            f"publication hygiene: {REPO_ROOT} is not a git checkout (or git is "
            "missing); the check must run inside one",
            file=sys.stderr,
        )
        return 2
    violations = (
        scan_filenames(tracked_files())
        + scan_content(paths)
        + scan_docs()
        + check_tracked_floor(paths)
    )
    try:
        session_ids = load_session_ids()
    except SessionIdSourceError as exc:
        session_ids = None
        violations.append(
            Violation("(session-id source)", 0, "session-id-source", str(exc))
        )
    else:
        if session_ids is None:
            print(SESSION_ID_SKIP_MESSAGE)
        else:
            violations += scan_session_ids(tracked_files(), session_ids)
            # Any other rule may echo a line that also holds a real ID.
            violations = [
                Violation(
                    redact_session_ids(v.path, session_ids),
                    v.line,
                    v.rule,
                    redact_session_ids(v.text, session_ids),
                )
                for v in violations
            ]

    if not args.quiet:
        for violation in violations:
            print(violation)

    print(
        f"publication hygiene: {len(paths)} tracked files, "
        f"{len(violations)} violation(s)"
    )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
