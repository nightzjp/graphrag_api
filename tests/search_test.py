"""
搜索测试脚本
"""

import argparse
from enum import Enum

from graphrag_api.search import SearchRunner

INVALID_METHOD_ERROR = "Invalid method"


class SearchType(Enum):
    """The type of search to run."""

    LOCAL = "local"
    GLOBAL = "global"
    DIRFT = "dirft"

    def __str__(self):
        """Return the string representation of the enum value."""
        return self.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        help="The configuration yaml file to use when running the query",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--data",
        help="The path with the output data from the pipeline",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--root",
        help="The data project root. Default value: the current directory",
        required=False,
        default=".",
        type=str,
    )

    parser.add_argument(
        "--method",
        help="The method to run, one of: local or global",
        required=True,
        type=SearchType,
        default="local",
        choices=list(SearchType),
    )

    parser.add_argument(
        "--community_level",
        help="Community level in the Leiden community hierarchy from which we will load the community reports higher value means we use reports on smaller communities",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--response_type",
        help="Free form text describing the response type and format, can be anything, e.g. Multiple Paragraphs, Single Paragraph, Single Sentence, List of 3-7 Points, Single Page, Multi-Page Report",
        type=str,
        default="Single Paragraph",
    )

    parser.add_argument(
        "query",
        nargs=1,
        help="The query to run",
        type=str,
    )

    args = parser.parse_args()

    search_runner = SearchRunner(
        config_filepath=args.config,
        data_dir=args.data,
        root_dir=args.root,
        community_level=args.community_level,
        response_type=args.response_type,
    )

    match args.method:
        case SearchType.LOCAL:
            search_runner.run_local_search(query=args.query[0], streaming=False)
        case SearchType.GLOBAL:
            search_runner.run_global_search(query=args.query[0], streaming=False)
        case SearchType.DIRFT:
            search_runner.run_direct_search(query=args.query[0])
        case _:
            raise ValueError(INVALID_METHOD_ERROR)
