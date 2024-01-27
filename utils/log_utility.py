import logging


def toggle_logging(enabled: bool = True):
    if enabled:
        logging.disable(logging.NOTSET)
        logging.basicConfig(filename=None, level=logging.INFO)
        logging.getLogger().addHandler(logging.NullHandler())  # logs to console
        logging.getLogger().setLevel(logging.INFO)
        logging.info("Logging enabled")

    else:
        logging.disable(logging.INFO)
        logging.disable(logging.CRITICAL)
        logging.disable(logging.WARNING)
