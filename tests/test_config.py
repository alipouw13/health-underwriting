
from app.config import load_settings, validate_settings


def test_load_settings_smoke():
    settings = load_settings()
    assert settings is not None
    assert hasattr(settings, "content_understanding")
    assert hasattr(settings, "openai")
    assert hasattr(settings, "app")


def test_validate_settings_returns_list():
    settings = load_settings()
    errors = validate_settings(settings)
    assert isinstance(errors, list)
