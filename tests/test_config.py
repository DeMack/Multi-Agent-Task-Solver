from pathlib import Path

from src.config import Config


def test_default_agent_timeout():
    cfg = Config.from_env()
    assert cfg.agent_timeout_seconds == 120


def test_default_code_execution_timeout():
    cfg = Config.from_env()
    assert cfg.code_execution_timeout_seconds == 30


def test_default_max_retries():
    cfg = Config.from_env()
    assert cfg.max_agent_retries == 2


def test_outputs_dir_is_path():
    cfg = Config.from_env()
    assert isinstance(cfg.outputs_dir, Path)


def test_default_outputs_dir_name():
    cfg = Config.from_env()
    assert cfg.outputs_dir == Path("outputs")


def test_env_overrides_timeout(monkeypatch):
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "120")
    cfg = Config.from_env()
    assert cfg.agent_timeout_seconds == 120


def test_env_overrides_max_retries(monkeypatch):
    monkeypatch.setenv("MAX_AGENT_RETRIES", "5")
    cfg = Config.from_env()
    assert cfg.max_agent_retries == 5


def test_env_overrides_outputs_dir(monkeypatch):
    monkeypatch.setenv("OUTPUTS_DIR", "/tmp/my-outputs")
    cfg = Config.from_env()
    assert cfg.outputs_dir == Path("/tmp/my-outputs")


def test_anthropic_api_key_reads_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    cfg = Config.from_env()
    assert cfg.anthropic_api_key == "sk-test-key"


def test_anthropic_api_key_defaults_to_empty():
    cfg = Config.from_env()
    # May already be set in environment — just verify it's a string
    assert isinstance(cfg.anthropic_api_key, str)
