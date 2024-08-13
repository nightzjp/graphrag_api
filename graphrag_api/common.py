import yaml
import json
from pathlib import Path
from typing import Optional

from graphrag.config import create_graphrag_config
from graphrag.index.progress import ProgressReporter


class BaseGraph:
    @staticmethod
    def _read_config_parameters(
        root: str, config: Optional[str], reporter: ProgressReporter
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
