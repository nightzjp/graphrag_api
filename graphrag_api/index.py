"""
Copyright (c) 2024 nightzjp.
Licensed under the MIT License

基于 graphrag\\index\\cli.py 修改
"""

import asyncio
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

from graphrag.config import CacheType, enable_logging_with_config, load_config, resolve_paths

from graphrag.index.validate_config import validate_config_names
from graphrag.index.api import build_index
from graphrag.index.emit.types import TableEmitterType
from graphrag.index.progress import ProgressReporter, ReporterType
from graphrag.index.progress.load_progress_reporter import load_progress_reporter

from graphrag.index.graph.extractors.claims.prompts import CLAIM_EXTRACTION_PROMPT
from graphrag.index.graph.extractors.community_reports.prompts import (
    COMMUNITY_REPORT_PROMPT,
)
from graphrag.index.graph.extractors.graph.prompts import GRAPH_EXTRACTION_PROMPT
from graphrag.index.graph.extractors.summarize.prompts import SUMMARIZE_PROMPT
from graphrag.index.init_content import INIT_DOTENV, INIT_YAML

from graphrag_api.common import BaseGraph

# Ignore warnings from numba
warnings.filterwarnings("ignore", message=".*NumbaDeprecationWarning.*")

log = logging.getLogger(__name__)


class GraphRagIndexer(BaseGraph):
    def __init__(
        self,
        root: str = ".",
        verbose: bool = False,
        resume: Optional[str] = None,
        update_index_id: Optional[str] = None,
        memprofile: bool = False,
        nocache: bool = False,
        reporter: ReporterType = ReporterType.RICH,
        config_filepath: Optional[str] = "",
        emit: list[TableEmitterType] = None,
        dryrun: bool = False,
        init: bool = False,
        skip_validations: bool = False,
        output_dir: Optional[str] = None,
    ):
        self.root = root
        self.verbose = verbose
        self.resume = resume
        self.update_index_id = update_index_id
        self.memprofile = memprofile
        self.nocache = nocache
        self.reporter = reporter
        self.config_filepath = config_filepath
        self.emit = emit
        self.dryrun = dryrun
        self.init = init
        self.skip_validations = skip_validations
        self.output_dir = output_dir
        self.cli = False

    @staticmethod
    def register_signal_handlers(reporter: ProgressReporter):
        import signal

        def handle_signal(signum, _):
            # Handle the signal here
            reporter.info(f"Received signal {signum}, exiting...")
            reporter.dispose()
            for task in asyncio.all_tasks():
                task.cancel()
            reporter.info("All tasks cancelled. Exiting...")

        # Register signal handlers for SIGINT and SIGHUP
        signal.signal(signal.SIGINT, handle_signal)

        if sys.platform != "win32":
            signal.signal(signal.SIGHUP, handle_signal)

    @staticmethod
    def logger(reporter: ProgressReporter):
        def info(msg: str, verbose: bool = False):
            log.info(msg)
            if verbose:
                reporter.info(msg)

        def error(msg: str, verbose: bool = False):
            log.error(msg)
            if verbose:
                reporter.error(msg)

        def success(msg: str, verbose: bool = False):
            log.info(msg)
            if verbose:
                reporter.success(msg)

        return info, error, success

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
                        result[key] = "==== REDACTED ===="
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
        progress_reporter = load_progress_reporter(self.reporter)
        info, error, success = self.logger(progress_reporter)
        run_id = self.resume or self.update_index_id or time.strftime("%Y%m%d-%H%M%S")

        if self.init:
            self._initialize_project_at(self.root, progress_reporter)
            sys.exit(0)

        root = Path(self.root).resolve()
        config = load_config(root, self.config_filepath)

        config.storage.base_dir = self.output_dir or config.storage.base_dir
        config.reporting.base_dir = self.output_dir or config.reporting.base_dir
        resolve_paths(config, run_id)

        if self.nocache:
            config.cache.type = CacheType.none

        enabled_logging, log_path = enable_logging_with_config(
            config, self.verbose
        )
        if enabled_logging:
            info(f"Logging enabled at {log_path}", True)
        else:
            info(
                f"Logging not enabled for config {self.redact(config.model_dump())}",
                True,
            )

        if self.skip_validations:
            validate_config_names(progress_reporter, config)
        info(f"Starting pipeline run for: {run_id}, {self.dryrun=}", self.verbose)
        info(
            f"Using default configuration: {self.redact(config.model_dump())}",
            self.verbose,
        )

        if self.dryrun:
            info("Dry run complete, exiting...", True)
            sys.exit(0)

        self.register_signal_handlers(progress_reporter)

        outputs = asyncio.run(
            build_index(
                config=config,
                run_id=run_id,
                is_resume_run=bool(self.resume),
                is_update_run=bool(self.update_index_id),
                memory_profile=self.memprofile,
                progress_reporter=progress_reporter,
                emit=self.emit
            )
        )

        encountered_errors = any(
            output.errors and len(output.errors) > 0 for output in outputs
        )

        progress_reporter.stop()
        if encountered_errors:
            error(
                "Errors occurred during the pipeline run, see logs for more details.", True
            )
        else:
            success("All workflows completed successfully.", True)

        sys.exit(1 if encountered_errors else 0)

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

    @staticmethod
    def _enable_logging(root: str, run_id: str, verbose: bool) -> None:
        """Enable logging to file and console."""
        log_file = Path(root) / "output" / run_id / "reports" / "indexing-engine.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        log_file.touch(exist_ok=True)

        logging.basicConfig(
            filename=str(log_file),
            filemode="a",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.DEBUG if verbose else logging.INFO,
        )

