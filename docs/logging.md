# Logging

There is two ways to set the logging level you can either import the `setup_console_logging` from `protoplast.utils`
or if you import the trainer module this is automatically call for you and the logging level will be pass to each
Ray processes.

You can configure the log level via the `LOG_LEVEL` environment variable by default the level of `INFO` is used.
This link show the supported level https://docs.python.org/3/library/logging.html#logging-levels