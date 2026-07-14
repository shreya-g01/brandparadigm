# brandparadigm.logging

`get_logger(name)` returns a standard-library `logging.Logger` configured once
(process-wide) with the project's format and the level from
`brandparadigm.config.settings.Settings.log_level`. Every module should call
`get_logger(__name__)` instead of configuring `logging` itself, so log output
stays consistent across scripts, the API, and the dashboard.
