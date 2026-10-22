import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for the backend application."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger("hmm_net")
