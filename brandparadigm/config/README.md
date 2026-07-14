# brandparadigm.config

Central configuration for the whole project.

- `paths.py` — canonical filesystem locations (`raw_data/`, `models/`, `configs/`, `docs/`), resolved relative to the repo root so every module agrees on where things live regardless of the current working directory.
- `settings.py` — a `pydantic-settings` `Settings` class that reads environment variables (and `.env`, see `.env.sample`) for anything that should be configurable per-environment: log level, model artifact paths, API host/port, dashboard base URL. Use `get_settings()` (cached) rather than instantiating `Settings()` directly or reading `os.environ` elsewhere in the codebase.
