from graphrag_api.index import GraphRagIndexer
from graphrag_api.search import SearchRunner


# 目录初始化

# indexer = GraphRagIndexer(root="rag", init=True)
#
# indexer.run()

# 索引初始化

# indexer = GraphRagIndexer(root="rag")
#
# indexer.run()

# 搜索测试

search_runner = SearchRunner(root_dir="rag")

print(search_runner.remove_sources(search_runner.run_local_search(query="你是谁?")))
# search_runner.run_local_search(query="你是谁?")
# search_runner.run_local_search(query="你是谁?")
# search_runner.run_global_search(query="你是谁?")
# search_runner.run_global_search(query="你是谁?")
# search_runner.run_global_search(query="你是谁?")
# search_runner.run_global_search(query="你是谁?")
