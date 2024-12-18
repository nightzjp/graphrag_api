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

from graphrag.api import build_index
from graphrag.config.init_content import INIT_DOTENV, INIT_YAML
from graphrag.config.load_config import load_config
from graphrag.config.logging import enable_logging_with_config
from graphrag.config.resolve_path import resolve_paths
from graphrag.config.enums import CacheType
from graphrag.logging.factory import create_progress_reporter
from graphrag.logging.types import ReporterType
from graphrag.prompts.index.entity_extraction import GRAPH_EXTRACTION_PROMPT
from graphrag.prompts.index.summarize_descriptions import SUMMARIZE_PROMPT
from graphrag.prompts.index.claim_extraction import CLAIM_EXTRACTION_PROMPT
from graphrag.prompts.index.community_report import COMMUNITY_REPORT_PROMPT
from graphrag.prompts.query.drift_search_system_prompt import DRIFT_LOCAL_SYSTEM_PROMPT
from graphrag.prompts.query.global_search_map_system_prompt import MAP_SYSTEM_PROMPT
from graphrag.prompts.query.global_search_reduce_system_prompt import (
    REDUCE_SYSTEM_PROMPT,
)
from graphrag.prompts.query.global_search_knowledge_system_prompt import (
    GENERAL_KNOWLEDGE_INSTRUCTION,
)
from graphrag.prompts.query.local_search_system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
from graphrag.prompts.query.question_gen_system_prompt import QUESTION_SYSTEM_PROMPT

from graphrag.index.validate_config import validate_config_names
from graphrag.logging.base import ProgressReporter

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
        dryrun: bool = False,
        init: bool = False,
        skip_validations: bool = False,
        output_dir: Optional[str] = None,
    ):
        self.root = root
        self.verbose = verbose
        self.resume = resume  # False表示修改
        self.update_index_id = update_index_id
        self.memprofile = memprofile
        self.nocache = nocache
        self.reporter = reporter
        self.config_filepath = config_filepath
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

    def run(self, is_updated=False):
        """Run the pipeline with the given config."""
        progress_reporter = create_progress_reporter(self.reporter)
        info, error, success = self.logger(progress_reporter)
        run_id = self.resume or time.strftime("%Y%m%d-%H%M%S")

        if self.init:  # 初始化
            self._initialize_project_at(self.root, progress_reporter)
            sys.exit(0)

        root = Path(self.root).resolve()
        config = load_config(root, self.config_filepath)

        if is_updated:
            if not config.update_index_storage:
                from graphrag.config.defaults import (
                    STORAGE_TYPE,
                    UPDATE_STORAGE_BASE_DIR,
                )
                from graphrag.config.models.storage_config import StorageConfig

                config.update_index_storage = StorageConfig(
                    type=STORAGE_TYPE,
                    base_dir=UPDATE_STORAGE_BASE_DIR,
                )

        config.storage.base_dir = (
            str(self.output_dir) if self.output_dir else config.storage.base_dir
        )
        config.reporting.base_dir = (
            str(self.output_dir) if self.output_dir else config.reporting.base_dir
        )
        resolve_paths(config, run_id)

        if self.nocache:
            config.cache.type = CacheType.none

        enabled_logging, log_path = enable_logging_with_config(config, self.verbose)
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
                memory_profile=self.memprofile,
                progress_reporter=progress_reporter,
            )
        )

        encountered_errors = any(
            output.errors and len(output.errors) > 0 for output in outputs
        )

        progress_reporter.stop()
        if encountered_errors:
            error(
                "Errors occurred during the pipeline run, see logs for more details.",
                True,
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

        with settings_yaml.open("wb") as file:
            file.write(INIT_YAML.encode(encoding="utf-8", errors="strict"))

        dotenv = root / ".env"
        if not dotenv.exists():
            with dotenv.open("wb") as file:
                file.write(INIT_DOTENV.encode(encoding="utf-8", errors="strict"))

        prompts_dir = root / "prompts"
        if not prompts_dir.exists():
            prompts_dir.mkdir(parents=True, exist_ok=True)

        prompts = {
            "entity_extraction": GRAPH_EXTRACTION_PROMPT,
            "summarize_descriptions": SUMMARIZE_PROMPT,
            "claim_extraction": CLAIM_EXTRACTION_PROMPT,
            "community_report": COMMUNITY_REPORT_PROMPT,
            "drift_search_system_prompt": DRIFT_LOCAL_SYSTEM_PROMPT,
            "global_search_map_system_prompt": MAP_SYSTEM_PROMPT,
            "global_search_reduce_system_prompt": REDUCE_SYSTEM_PROMPT,
            "global_search_knowledge_system_prompt": GENERAL_KNOWLEDGE_INSTRUCTION,
            "local_search_system_prompt": LOCAL_SEARCH_SYSTEM_PROMPT,
            "question_gen_system_prompt": QUESTION_SYSTEM_PROMPT,
        }

        for name, content in prompts.items():
            prompt_file = prompts_dir / f"{name}.txt"
            if not prompt_file.exists():
                with prompt_file.open("wb") as file:
                    file.write(content.encode(encoding="utf-8", errors="strict"))

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
