"""
Copyright (c) 2024 nightzjp.
Licensed under the MIT License

基于 graphrag\\index\\cli.py 修改
"""

import asyncio
import json
import yaml
import logging
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Union

from graphrag.config import create_graphrag_config
from graphrag.index import PipelineConfig, create_pipeline_config
from graphrag.index.cache import NoopPipelineCache
from graphrag.index.progress import (
    NullProgressReporter,
    PrintProgressReporter,
    ProgressReporter,
)
from graphrag.index.progress.rich import RichProgressReporter
from graphrag.index.run import run_pipeline_with_config

from graphrag.index.emit import TableEmitterType
from graphrag.index.graph.extractors.claims.prompts import CLAIM_EXTRACTION_PROMPT
from graphrag.index.graph.extractors.community_reports.prompts import (
    COMMUNITY_REPORT_PROMPT,
)
from graphrag.index.graph.extractors.graph.prompts import GRAPH_EXTRACTION_PROMPT
from graphrag.index.graph.extractors.summarize.prompts import SUMMARIZE_PROMPT
from graphrag.index.init_content import INIT_DOTENV, INIT_YAML

# Ignore warnings from numba
warnings.filterwarnings("ignore", message=".*NumbaDeprecationWarning.*")

log = logging.getLogger(__name__)


class GraphRagIndexer:
    def __init__(
        self,
        root: str = ".",
        verbose: bool = False,
        resume: Optional[str] = None,
        memprofile: bool = False,
        nocache: bool = False,
        reporter: Optional[str] = "",
        config: Optional[str] = "",
        emit: Optional[str] = "",
        dryrun: bool = False,
        init: bool = False,
        overlay_defaults: bool = False,
    ):
        self.root = root
        self.verbose = verbose
        self.resume = resume
        self.memprofile = memprofile
        self.nocache = nocache
        self.reporter = reporter
        self.config = config
        self.emit = emit
        self.dryrun = dryrun
        self.init = init
        self.overlay_defaults = overlay_defaults

    @staticmethod
    def redact(input: dict) -> str:
        """Sanitize the config json."""

        def redact_dict(input: dict) -> dict:
            if not isinstance(input, dict):
                return input

            result = {}
            for key, value in input.items():
                if key in {
                    "api_key",
                    "connection_string",
                    "container_name",
                    "organization",
                }:
                    if value is not None:
                        result[key] = f"REDACTED, length {len(value)}"
                elif isinstance(value, dict):
                    result[key] = redact_dict(value)
                elif isinstance(value, list):
                    result[key] = [redact_dict(i) for i in value]
                else:
                    result[key] = value
            return result

        redacted_dict = redact_dict(input)
        return json.dumps(redacted_dict, indent=4)

    def run(self):
        """Run the pipeline with the given config."""
        run_id = self.resume or time.strftime("%Y%m%d-%H%M%S")
        self._enable_logging(self.root, run_id, self.verbose)
        progress_reporter = self._get_progress_reporter(self.reporter)
        if self.init:
            self._initialize_project_at(self.root, progress_reporter)
            sys.exit(0)
        if self.overlay_defaults:
            pipeline_config: Union[str, PipelineConfig] = self._create_default_config(
                self.root,
                self.config,
                self.verbose,
                self.dryrun or False,
                progress_reporter,
            )
        else:
            pipeline_config: Union[
                str, PipelineConfig
            ] = self.config or self._create_default_config(
                self.root, None, self.verbose, self.dryrun or False, progress_reporter
            )
        cache = NoopPipelineCache() if self.nocache else None
        pipeline_emit = self.emit.split(",") if self.emit else None
        encountered_errors = False

        def _run_workflow_async() -> None:
            import signal

            def handle_signal(signum, _):
                # Handle the signal here
                progress_reporter.info(f"Received signal {signum}, exiting...")
                progress_reporter.dispose()
                for task in asyncio.all_tasks():
                    task.cancel()
                progress_reporter.info("All tasks cancelled. Exiting...")

            # Register signal handlers for SIGINT and SIGHUP
            signal.signal(signal.SIGINT, handle_signal)

            if sys.platform != "win32":
                signal.signal(signal.SIGHUP, handle_signal)

            async def execute():
                nonlocal encountered_errors
                async for output in run_pipeline_with_config(
                    pipeline_config,
                    run_id=run_id,
                    memory_profile=self.memprofile,
                    cache=cache,
                    progress_reporter=progress_reporter,
                    emit=(
                        [TableEmitterType(e) for e in pipeline_emit]
                        if pipeline_emit
                        else None
                    ),
                    is_resume_run=bool(self.resume),
                ):
                    if output.errors and len(output.errors) > 0:
                        encountered_errors = True
                        progress_reporter.error(output.workflow)
                    else:
                        progress_reporter.success(output.workflow)

                    progress_reporter.info(str(output.result))

            if platform.system() == "Windows":
                import nest_asyncio  # type: ignore Ignoring because out of windows this will cause an error

                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                loop.run_until_complete(execute())
            elif sys.version_info >= (3, 11):
                import uvloop  # type: ignore Ignoring because on windows this will cause an error

                with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:  # type: ignore Ignoring because minor versions this will throw an error
                    runner.run(execute())
            else:
                import uvloop  # type: ignore Ignoring because on windows this will cause an error

                uvloop.install()
                asyncio.run(execute())

        _run_workflow_async()
        progress_reporter.stop()
        if encountered_errors:
            progress_reporter.error(
                "Errors occurred during the pipeline run, see logs for more details."
            )
        else:
            progress_reporter.success("All workflows completed successfully.")

    @staticmethod
    def _initialize_project_at(path: str, reporter: ProgressReporter) -> None:
        """Initialize the project at the given path."""
        reporter.info(f"Initializing project at {path}")
        root = Path(path)
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)

        settings_yaml = root / "settings.yaml"
        if settings_yaml.exists():
            msg = f"Project already initialized at {root}"
            raise ValueError(msg)

        dotenv = root / ".env"
        if not dotenv.exists():
            with settings_yaml.open("wb") as file:
                file.write(INIT_YAML.encode(encoding="utf-8", errors="strict"))

        with dotenv.open("wb") as file:
            file.write(INIT_DOTENV.encode(encoding="utf-8", errors="strict"))

        prompts_dir = root / "prompts"
        if not prompts_dir.exists():
            prompts_dir.mkdir(parents=True, exist_ok=True)

        entity_extraction = prompts_dir / "entity_extraction.txt"
        if not entity_extraction.exists():
            with entity_extraction.open("wb") as file:
                file.write(
                    GRAPH_EXTRACTION_PROMPT.encode(encoding="utf-8", errors="strict")
                )

        summarize_descriptions = prompts_dir / "summarize_descriptions.txt"
        if not summarize_descriptions.exists():
            with summarize_descriptions.open("wb") as file:
                file.write(SUMMARIZE_PROMPT.encode(encoding="utf-8", errors="strict"))

        claim_extraction = prompts_dir / "claim_extraction.txt"
        if not claim_extraction.exists():
            with claim_extraction.open("wb") as file:
                file.write(
                    CLAIM_EXTRACTION_PROMPT.encode(encoding="utf-8", errors="strict")
                )

        community_report = prompts_dir / "community_report.txt"
        if not community_report.exists():
            with community_report.open("wb") as file:
                file.write(
                    COMMUNITY_REPORT_PROMPT.encode(encoding="utf-8", errors="strict")
                )

    def _create_default_config(
        self,
        root: str,
        config: Optional[str],
        verbose: bool,
        dryrun: bool,
        reporter: ProgressReporter,
    ) -> PipelineConfig:
        """Overlay default values on an existing config or create a default config if none is provided."""
        if config and not Path(config).exists():
            msg = f"Configuration file {config} does not exist"
            raise ValueError(msg)

        if not Path(root).exists():
            msg = f"Root directory {root} does not exist"
            raise ValueError(msg)

        parameters = self._read_config_parameters(root, config, reporter)
        log.info(
            "using default configuration: %s",
            self.redact(parameters.model_dump()),
        )

        if verbose or dryrun:
            reporter.info(
                f"Using default configuration: {self.redact(parameters.model_dump())}"
            )
        result = create_pipeline_config(parameters, verbose)
        if verbose or dryrun:
            reporter.info(f"Final Config: {self.redact(result.model_dump())}")

        if dryrun:
            reporter.info("dry run complete, exiting...")
            sys.exit(0)
        return result

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
                data = yaml.safe_load(file.read().decode(encoding="utf-8", errors="strict"))
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
    def _enable_logging(root: str, run_id: str, verbose: bool) -> None:
        """Enable logging to file and console."""
        root = Path(root)
        log_file = root / "logs" / f"{run_id}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=level, format=fmt)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)
        log.info("logging enabled")

    @staticmethod
    def _get_progress_reporter(
        progress_reporter: Optional[str] = None,
    ) -> ProgressReporter:
        """Enable progress reporting to console."""
        _reporter = progress_reporter or "print"
        if _reporter == "null":
            return NullProgressReporter()
        elif _reporter == "print":
            return PrintProgressReporter("GraphRAG Indexer ")
        elif _reporter == "rich":
            return RichProgressReporter("GraphRAG Indexer ")
        else:
            msg = (
                f"Unsupported progress reporter: {_reporter}. "
                f"Supported reporters are null, print and rich"
            )
            raise ValueError(msg)
