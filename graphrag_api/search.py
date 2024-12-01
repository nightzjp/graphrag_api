"""
Copyright (c) 2024 nightzjp.
Licensed under the MIT License

基于 graphrag\\query\\cli.py 修改
"""
import re
import sys
import asyncio
from pathlib import Path
from collections.abc import AsyncGenerator

import pandas as pd
from pydantic import validate_call

from graphrag.config import GraphRagConfig, load_config, resolve_paths
from graphrag.index.progress import PrintProgressReporter
from graphrag.index.create_pipeline_config import create_pipeline_config
from graphrag.utils.storage import _create_storage, _load_table_from_storage
from graphrag.model.entity import Entity
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.structured_search.base import SearchResult
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
        config_filepath=None,
        data_dir=None,
        root_dir="rag",
        community_level=2,
        response_type="Single Paragraph",
    ):
        self.config_filepath = config_filepath
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
        result: SearchResult = await search_agent.asearch(query=query)
        response = result.response
        context_data = _reformat_context_data(result.context_data)  # type: ignore
        return response, context_data

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
        root = Path(self.root_dir).resolve()
        config = load_config(root, self.config_filepath)

        config.storage.base_dir = self.data_dir or config.storage.base_dir
        resolve_paths(config)

        dataframe_dict = self._resolve_parquet_files(
            root_dir=self.root_dir,
            config=config,
            parquet_list=[
                "create_final_nodes.parquet",
                "create_final_entities.parquet",
                "create_final_community_reports.parquet",
            ],
            optional_list=[]
        )

        final_nodes: pd.DataFrame = dataframe_dict["create_final_nodes"]
        final_entities: pd.DataFrame = dataframe_dict["create_final_entities"]
        final_community_reports: pd.DataFrame = dataframe_dict[
            "create_final_community_reports"
        ]

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
        root = Path(self.root_dir).resolve()
        config = load_config(root, self.config_filepath)

        config.storage.base_dir = self.data_dir or config.storage.base_dir
        resolve_paths(config)

        dataframe_dict = self._resolve_parquet_files(
            root_dir=self.root_dir,
            config=config,
            parquet_list=[
                "create_final_nodes.parquet",
                "create_final_community_reports.parquet",
                "create_final_text_units.parquet",
                "create_final_relationships.parquet",
                "create_final_entities.parquet",
            ],
            optional_list=["create_final_covariates.parquet"]
        )

        final_nodes: pd.DataFrame = dataframe_dict["create_final_nodes"]
        final_community_reports: pd.DataFrame = dataframe_dict[
            "create_final_community_reports"
        ]
        final_text_units: pd.DataFrame = dataframe_dict["create_final_text_units"]
        final_relationships: pd.DataFrame = dataframe_dict["create_final_relationships"]
        final_entities: pd.DataFrame = dataframe_dict["create_final_entities"]
        final_covariates: pd.DataFrame | None = dataframe_dict["create_final_covariates"]

        vector_store_args = (
            config.embeddings.vector_store if config.embeddings.vector_store else {}
        )

        reporter.info(f"Vector Store Args: {vector_store_args}")
        vector_store_type = vector_store_args.get("type", VectorStoreType.LanceDB)

        entities = read_indexer_entities(
            final_nodes, final_entities, self.community_level
        )

        lancedb_dir = Path(config.storage.base_dir) / "lancedb"

        vector_store_args.update({"db_uri": str(lancedb_dir)})
        description_embedding_store = self.__get_embedding_description_store(
            entities=entities,
            vector_store_type=vector_store_type,
            config_args=vector_store_args,
        )
        covariates = read_indexer_covariates(final_covariates) if final_covariates is not None else []

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

    @staticmethod
    def _resolve_parquet_files(
            root_dir: str,
            config: GraphRagConfig,
            parquet_list: list[str],
            optional_list: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Read parquet files to a dataframe dict."""
        dataframe_dict = {}
        pipeline_config = create_pipeline_config(config)
        storage_obj = _create_storage(root_dir=root_dir, config=pipeline_config.storage)
        for parquet_file in parquet_list:
            df_key = parquet_file.split(".")[0]
            df_value = asyncio.run(
                _load_table_from_storage(name=parquet_file, storage=storage_obj)
            )
            dataframe_dict[df_key] = df_value

        # for optional parquet files, set the dict entry to None instead of erroring out if it does not exist
        for optional_file in optional_list:
            file_exists = asyncio.run(storage_obj.has(optional_file))
            df_key = optional_file.split(".")[0]
            if file_exists:
                df_value = asyncio.run(
                    _load_table_from_storage(name=optional_file, storage=storage_obj)
                )
                dataframe_dict[df_key] = df_value
            else:
                dataframe_dict[df_key] = None
        return dataframe_dict

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
