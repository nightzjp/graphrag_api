"""
Copyright (c) 2024 nightzjp.
Licensed under the MIT License

基于 graphrag\\query\\cli.py 修改
"""
import asyncio
import re
import sys
from pathlib import Path
from typing import cast
from pydantic import validate_call
from collections.abc import AsyncGenerator

import pandas as pd

from graphrag.config import GraphRagConfig
from graphrag.index.progress import PrintProgressReporter
from graphrag.model.entity import Entity
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.vector_stores import VectorStoreFactory, VectorStoreType
from graphrag.vector_stores.lancedb import LanceDBVectorStore

from graphrag.query.factories import get_global_search_engine, get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.api import _reformat_context_data

from graphrag_api.common import BaseGraph

reporter = PrintProgressReporter("")


class SearchRunner(BaseGraph):
    def __init__(
        self,
        config_dir=None,
        data_dir=None,
        root_dir="rag",
        community_level=2,
        response_type="Single Paragraph",
    ):
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.root_dir = root_dir
        self.community_level = community_level
        self.response_type = response_type
        self.config = None
        self.__local_agent = self.__get__local_agent()
        self.__global_agent = self.__get__global_agent()

    @staticmethod
    @validate_call(config={"arbitrary_types_allowed": True})
    async def search(search_agent, query):
        """非流式搜索"""
        result = await search_agent.asearch(query=query)
        return result.response

    @staticmethod
    @validate_call(config={"arbitrary_types_allowed": True})
    async def search_streaming(search_agent, query) -> AsyncGenerator:
        """流式搜索"""
        search_result = search_agent.astream_search(query=query)
        context_data = None
        get_context_data = True
        async for stream_chunk in search_result:
            if get_context_data:
                context_data = _reformat_context_data(stream_chunk)
                yield context_data
                get_context_data = False
            else:
                yield stream_chunk

    async def run_streaming_search(self, search_agent, query):
        full_response = ""
        context_data = None
        get_context_data = True
        async for stream_chunk in self.search_streaming(
            search_agent=search_agent, query=query
        ):
            if get_context_data:
                context_data = stream_chunk
                get_context_data = False
            else:
                full_response += stream_chunk
                print(stream_chunk, end="")  # noqa: T201
                sys.stdout.flush()  # flush output buffer to display text immediately
        print()  # noqa: T201
        return full_response, context_data

    @staticmethod
    def __get_embedding_description_store(
        entities: list[Entity],
        vector_store_type: str = VectorStoreType.LanceDB,
        config_args: dict | None = None,
    ):
        """Get the embedding description store."""
        if not config_args:
            config_args = {}

        collection_name = config_args.get(
            "query_collection_name", "entity_description_embeddings"
        )
        config_args.update({"collection_name": collection_name})
        description_embedding_store = VectorStoreFactory.get_vector_store(
            vector_store_type=vector_store_type, kwargs=config_args
        )

        description_embedding_store.connect(**config_args)

        if config_args.get("overwrite", True):
            # this step assumps the embeddings where originally stored in a file rather
            # than a vector database

            # dump embeddings from the entities list to the description_embedding_store
            store_entity_semantic_embeddings(
                entities=entities, vectorstore=description_embedding_store
            )
        else:
            # load description embeddings to an in-memory lancedb vectorstore
            # to connect to a remote db, specify url and port values.
            description_embedding_store = LanceDBVectorStore(
                collection_name=collection_name
            )
            description_embedding_store.connect(
                db_uri=config_args.get("db_uri", "./lancedb")
            )

            # load data from an existing table
            description_embedding_store.document_collection = (
                description_embedding_store.db_connection.open_table(
                    description_embedding_store.collection_name
                )
            )

        return description_embedding_store

    def __get__global_agent(self):
        """获取global搜索引擎"""
        data_dir, root_dir, config = self._configure_paths_and_settings(
            self.data_dir, self.root_dir, self.config_dir
        )
        data_path = Path(data_dir)

        final_nodes: pd.DataFrame = pd.read_parquet(
            data_path / "create_final_nodes.parquet"
        )
        final_entities: pd.DataFrame = pd.read_parquet(
            data_path / "create_final_entities.parquet"
        )
        final_community_reports: pd.DataFrame = pd.read_parquet(
            data_path / "create_final_community_reports.parquet"
        )

        reports = read_indexer_reports(
            final_community_reports, final_nodes, self.community_level
        )
        entities = read_indexer_entities(
            final_nodes, final_entities, self.community_level
        )
        return get_global_search_engine(
            config,
            reports=reports,
            entities=entities,
            response_type=self.response_type,
        )

    def run_global_search(self, query, streaming=False):
        """Run a global search with the given query."""

        if streaming:
            return asyncio.run(
                self.run_streaming_search(search_agent=self.__global_agent, query=query)
            )

        return asyncio.run(self.search(search_agent=self.__global_agent, query=query))

    def __get__local_agent(self):
        """获取local搜索引擎"""
        data_dir, root_dir, config = self._configure_paths_and_settings(
            self.data_dir, self.root_dir, self.config_dir
        )
        data_path = Path(data_dir)

        final_nodes = pd.read_parquet(data_path / "create_final_nodes.parquet")
        final_community_reports = pd.read_parquet(
            data_path / "create_final_community_reports.parquet"
        )
        final_text_units = pd.read_parquet(
            data_path / "create_final_text_units.parquet"
        )
        final_relationships = pd.read_parquet(
            data_path / "create_final_relationships.parquet"
        )
        final_entities = pd.read_parquet(data_path / "create_final_entities.parquet")
        final_covariates_path = data_path / "create_final_covariates.parquet"
        final_covariates = (
            pd.read_parquet(final_covariates_path)
            if final_covariates_path.exists()
            else None
        )

        vector_store_args = (
            config.embeddings.vector_store if config.embeddings.vector_store else {}
        )

        reporter.info(f"Vector Store Args: {vector_store_args}")
        vector_store_type = vector_store_args.get("type", VectorStoreType.LanceDB)

        entities = read_indexer_entities(
            final_nodes, final_entities, self.community_level
        )
        description_embedding_store = self.__get_embedding_description_store(
            entities=entities,
            vector_store_type=vector_store_type,
            config_args=vector_store_args,
        )
        covariates = (
            read_indexer_covariates(final_covariates)
            if final_covariates is not None
            else []
        )

        return get_local_search_engine(
            config,
            reports=read_indexer_reports(
                final_community_reports, final_nodes, self.community_level
            ),
            text_units=read_indexer_text_units(final_text_units),
            entities=entities,
            relationships=read_indexer_relationships(final_relationships),
            covariates={"claims": covariates},
            description_embedding_store=description_embedding_store,
            response_type=self.response_type,
        )

    def run_local_search(self, query, streaming=False):
        """Run a local search with the given query."""

        if streaming:
            return asyncio.run(
                self.run_streaming_search(search_agent=self.__local_agent, query=query)
            )
        return asyncio.run(self.search(search_agent=self.__local_agent, query=query))

    def _configure_paths_and_settings(
        self,
        data_dir: str | None,
        root_dir: str | None,
        config_dir: str | None,
    ) -> tuple[str, str | None, GraphRagConfig]:
        if data_dir is None and root_dir is None:
            msg = "Either data_dir or root_dir must be provided."
            raise ValueError(msg)
        if data_dir is None:
            data_dir = self._infer_data_dir(cast(str, root_dir))
        config = self._create_graphrag_config(root_dir, config_dir)
        return data_dir, root_dir, config

    @staticmethod
    def _infer_data_dir(root: str) -> str:
        output = Path(root) / "output"
        # use the latest data-run folder
        if output.exists():
            expr = re.compile(r"\d{8}-\d{6}")
            filtered = [
                f for f in output.iterdir() if f.is_dir() and expr.match(f.name)
            ]
            folders = sorted(filtered, key=lambda f: f.name, reverse=True)
            if len(folders) > 0:
                folder = folders[0]
                return str((folder / "artifacts").absolute())
        msg = f"Could not infer data directory from root={root}"
        raise ValueError(msg)

    def _create_graphrag_config(
        self,
        root: str | None,
        config_dir: str | None,
    ) -> GraphRagConfig:
        """Create a GraphRag configuration."""
        return self._read_config_parameters(root or "./", config_dir, reporter)

    @staticmethod
    def remove_sources(text):
        # 使用正则表达式匹配 [Data: Sources (82, 14, 42, 98)] 这种格式的字符串
        cleaned_text = re.sub(r'\[Data: [^]]+\]', '', text)
        return cleaned_text
