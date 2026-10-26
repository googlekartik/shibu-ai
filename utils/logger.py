"""Structured logging. Never logs tokens or other secrets."""
import logging
import time

_logger = None


def get_logger(name="rule_based_chatbot"):
    global _logger
    if _logger is not None:
        return _logger
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _logger = logger
    return logger


def log_chat(logger, intent, rule_id, confidence, tool, elapsed_ms, error=None):
    msg = (
        f"intent={intent} rule={rule_id} confidence={confidence:.2f} "
        f"tool={tool} elapsed_ms={elapsed_ms:.1f}"
    )
    if error:
        logger.warning(f"{msg} error={error}")
    else:
        logger.info(msg)


class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsed_ms(self):
        return (time.time() - self.start) * 1000.0
