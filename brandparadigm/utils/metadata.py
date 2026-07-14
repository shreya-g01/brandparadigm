"""Reusable run-metadata helpers: git commit hash, installed library versions.

Used by every model training pipeline (starting with
`brandparadigm.sentiment.train`) to record exactly what produced a given
set of artifacts — reused rather than reimplemented per model in later
phases (BERTopic, the topic classifier).
"""

import subprocess
from importlib.metadata import PackageNotFoundError, version

from brandparadigm.logging import get_logger

logger = get_logger(__name__)


def get_git_commit_hash() -> str | None:
    """Return the current git commit hash, or None if unavailable.

    None (rather than raising) when this isn't a git checkout, `git` isn't
    installed, or the command otherwise fails — metadata generation should
    never break a training run over a missing commit hash.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Could not determine git commit hash: %s", exc)
        return None


def get_library_versions(package_names: list[str]) -> dict[str, str | None]:
    """Return installed version strings for `package_names` (None if not installed)."""
    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions
