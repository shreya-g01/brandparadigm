import re

from brandparadigm.utils.metadata import get_git_commit_hash, get_library_versions


def test_get_git_commit_hash_returns_a_valid_sha_in_this_repo():
    commit_hash = get_git_commit_hash()
    assert commit_hash is not None
    assert re.fullmatch(r"[0-9a-f]{40}", commit_hash)


def test_get_library_versions_returns_known_installed_packages():
    versions = get_library_versions(["pytest", "pyyaml"])
    assert versions["pytest"] is not None
    assert re.match(r"^\d+\.\d+", versions["pytest"])


def test_get_library_versions_returns_none_for_unknown_package():
    versions = get_library_versions(["this-package-does-not-exist-xyz"])
    assert versions["this-package-does-not-exist-xyz"] is None


def test_get_library_versions_preserves_input_order_and_count():
    names = ["pytest", "this-package-does-not-exist-xyz", "pyyaml"]
    versions = get_library_versions(names)
    assert list(versions.keys()) == names
