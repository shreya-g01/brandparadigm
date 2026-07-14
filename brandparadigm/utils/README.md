# brandparadigm.utils

Small, dependency-light helpers shared by every other subpackage:

- `io.py` — `read_yaml`, `read_json`, `write_json`, `ensure_dir`. All config loading and artifact/report writing goes through these instead of ad-hoc `open()` calls, so path creation and JSON formatting stay consistent.
- `seed.py` — `set_seed(seed)` seeds Python, NumPy, and (if installed) PyTorch RNGs. Every training/evaluation script calls this first for reproducible runs.
- `metadata.py` — `get_git_commit_hash()` and `get_library_versions(names)`, used by every model training pipeline (starting with `brandparadigm.sentiment.train`) to write a self-describing `metadata.json` alongside saved model artifacts. Both fail soft (`None`/per-package `None`) rather than raising, so missing git or an unlisted package never breaks a training run.
