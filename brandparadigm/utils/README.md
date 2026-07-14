# brandparadigm.utils

Small, dependency-light helpers shared by every other subpackage:

- `io.py` — `read_yaml`, `read_json`, `write_json`, `ensure_dir`. All config loading and artifact/report writing goes through these instead of ad-hoc `open()` calls, so path creation and JSON formatting stay consistent.
- `seed.py` — `set_seed(seed)` seeds Python, NumPy, and (if installed) PyTorch RNGs. Every training/evaluation script calls this first for reproducible runs.
