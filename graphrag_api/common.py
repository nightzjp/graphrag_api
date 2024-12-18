import yaml
import json
from pathlib import Path
from typing import Optional

from graphrag.config.create_graphrag_config import create_graphrag_config

from graphrag.logger.base import ProgressLogger


class BaseGraph:
    @staticmethod
    def _read_config_parameters(
        root: str, config: Optional[str], reporter: ProgressLogger
    ):
        _root = Path(root)
        settings_yaml = (
            Path(config)
            if config and Path(config).suffix in [".yaml", ".yml"]
            else _root / "settings.yaml"
        )
        settings_json = (
            Path(config)
            if config and Path(config).suffix in [".json"]
            else _root / "settings.json"
        )
        if settings_yaml.exists():
            reporter.success(f"Reading settings from {settings_yaml}")
            with settings_yaml.open("rb") as file:
                data = yaml.safe_load(
                    file.read().decode(encoding="utf-8", errors="strict")
                )
                result = create_graphrag_config(data, root)
        elif settings_json.exists():
            reporter.success(f"Reading settings from {settings_json}")
            with settings_json.open("rb") as file:
                data = json.loads(file.read().decode(encoding="utf-8", errors="strict"))
                result = create_graphrag_config(data, root)
        else:
            msg = (
                f"Cannot find any configuration files at {settings_json} or "
                f"{settings_yaml}"
            )
            raise ValueError(msg)
        return result

    @staticmethod
    def redact(config: dict) -> str:
        """Sanitize secrets in a config object."""

        # Redact any sensitive configuration
        def redact_dict(config: dict) -> dict:
            if not isinstance(config, dict):
                return config

            result = {}
            for key, value in config.items():
                if key in {
                    "api_key",
                    "connection_string",
                    "container_name",
                    "organization",
                }:
                    if value is not None:
                        result[key] = "==== REDACTED ===="
                elif isinstance(value, dict):
                    result[key] = redact_dict(value)
                elif isinstance(value, list):
                    result[key] = [redact_dict(i) for i in value]
                else:
                    result[key] = value
            return result

        redacted_dict = redact_dict(config)
        return json.dumps(redacted_dict, indent=4)
