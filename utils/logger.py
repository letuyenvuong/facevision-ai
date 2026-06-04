import logging
import sys

RESET   = "\033[0m"
BOLD    = "\033[1m"
RED     = "\033[31m"
YELLOW  = "\033[33m"
GREEN   = "\033[32m"
CYAN    = "\033[36m"
MAGENTA = "\033[35m"

LEVEL_COLORS = {
    logging.DEBUG:    CYAN,
    logging.INFO:     GREEN,
    logging.WARNING:  YELLOW,
    logging.ERROR:    RED,
    logging.CRITICAL: MAGENTA + BOLD,
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = LEVEL_COLORS.get(record.levelno, RESET)
        record.levelname = f"{color}{record.levelname:<8}{RESET}"
        record.name = f"{CYAN}{record.name}{RESET}"
        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
