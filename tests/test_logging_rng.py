import logging
import random

import pytest
from utils.logging_config import setup_logging


@pytest.mark.parametrize("message", ["checkpoint saved", "Restored epoch=%s", "Loading /tmp/state.pt"])
def test_regular_logging_does_not_advance_python_rng(message):
    root = logging.getLogger()
    old_handlers, old_level = list(root.handlers), root.level
    original_rng = random.getstate()
    try:
        setup_logging(is_main_process=True, preserve_hydra_handlers=False)
        random.seed(42)
        before = random.getstate()
        if "%s" in message:
            logging.getLogger("resume-audit").info(message, 3)
        else:
            logging.getLogger("resume-audit").info(message)
        assert random.getstate() == before
    finally:
        root.handlers[:] = old_handlers
        root.setLevel(old_level)
        random.setstate(original_rng)
