import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    anthropic_api_key: str
    agent_timeout_seconds: int
    code_execution_timeout_seconds: int
    max_agent_retries: int
    outputs_dir: Path

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            agent_timeout_seconds=int(os.environ.get("AGENT_TIMEOUT_SECONDS", "60")),
            code_execution_timeout_seconds=int(
                os.environ.get("CODE_EXECUTION_TIMEOUT_SECONDS", "30")
            ),
            max_agent_retries=int(os.environ.get("MAX_AGENT_RETRIES", "2")),
            outputs_dir=Path(os.environ.get("OUTPUTS_DIR", "outputs")),
        )


config = Config.from_env()
