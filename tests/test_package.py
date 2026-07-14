import brandparadigm
from brandparadigm.config import get_settings
from brandparadigm.logging import get_logger
from brandparadigm.utils import set_seed


def test_package_version_is_set():
    assert brandparadigm.__version__


def test_settings_load_with_defaults():
    settings = get_settings()
    assert settings.api_port == 8000
    assert settings.log_level == "INFO"


def test_get_logger_returns_named_logger():
    logger = get_logger("brandparadigm.test")
    assert logger.name == "brandparadigm.test"


def test_set_seed_is_deterministic():
    import random

    set_seed(123)
    first = random.random()
    set_seed(123)
    second = random.random()
    assert first == second
