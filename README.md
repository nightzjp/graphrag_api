# GraphRag-API

GraphRag-API 在 GraphRag 库的基础上扩展，提供了 API 调用功能，用于执行本地和全局搜索。此扩展允许用户通过 RESTful API 调用，轻松地将 GraphRag 的强大搜索功能集成到他们的应用程序中。

## 功能

- **本地搜索**：在指定的本地数据集内进行搜索。
- **全局搜索**：在更广泛的全球数据集中进行搜索。
- **可配置参数**：通过配置文件或 API 调用自定义搜索参数。
- **RESTful API**：通过 HTTP 请求轻松与其他应用程序和服务集成。

## 安装

### 先决条件

确保已安装 Python 3.8+。

### 通过 pip 安装

使用 pip 安装 GraphRag-API：

```bash
pip install graphrag_api
```

### 从源码安装

1. 克隆源码库：

```bash
git clone https://github.com/nightzjp/graphrag_api
```

1. 进入项目目录并安装依赖：

```bash
cd graphrag_api
pip install -r requirements.txt
```

## 使用

### 初始化

1. 命令行初始化

```bash
python -m graphrag index --init --root ./rag  # graphrag初始化
python index_test.py --init --root rag  # graphrag_api初始化
```

2代码初始化

```python
from graphrag_api.index import GraphRagIndexer


indexer = GraphRagIndexer(root="rag", init=True)

indexer.run()
```

### 索引创建

1. 命令行初始化(会生成rag目录)

```bash
python -m graphrag index --root rag  # graphrag初始化
python index_test.py --root rag  # graphrag_api初始化
```

2. 代码初始化

```python
from graphrag_api.index import GraphRagIndexer


indexer = GraphRagIndexer(root="rag")

indexer.run()
```

3. 修改配置文件(自动生成，需要修改相应配置)

`.env`文件

```dotenv
GRAPHRAG_API_KEY=<API_KEY>
```

`settings.yaml`文件

```yaml

encoding_model: cl100k_base
skip_workflows: []
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat # or azure_openai_chat
  model: gpt-4o-mini  # mini性价比比较高
  model_supports_json: true # recommended if this is available for your model.

embeddings:
  ## parallelization: override the global parallelization settings for embeddings
  async_mode: threaded # or asyncio
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding # or azure_openai_embedding
    model: text-embedding-3-small
    
input:
  type: file # or blob
  file_type: csv # or text  这里以csv为例
  base_dir: "input"
  file_encoding: utf-8
  file_pattern: ".*\\.csv$"
  source_column: "question"  # csv-key
  text_column: "answer"  # csv-key

```

`q.csv`文件示例

```text
question,answer
"你是谁","你猜啊"
```

### 搜索

1. 命令行初始化

```bash
python -m graphrag query \
--root ./ragtest \
--method global(local) \
"What are the top themes in this story?"  # graphrag初始化

python search_test.py --root rag --method global(local) "What are the top themes in this story?"  # graphrag初始化
```

2代码初始化

```python
from graphrag_api.search import SearchRunner

search_runner = SearchRunner(root_dir="rag")

search_runner.run_local_search(query="What are the top themes in this story?", streaming=False)
search_runner.run_global_search(query="What are the top themes in this story?", streaming=False)

# 对于输出的结果可能带有一些特殊字符，可以采用以下函数去除特殊字符或自行处理。
search_runner.remove_sources(search_runner.run_local_search(query="What are the top themes in this story?")[0])
```

### 报告问题

如果遇到任何问题，请在 GitHub 上提交 issue。

## 许可证

此项目依据 MIT 许可证授权。有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

## 致谢

此项目基于 [GraphRag](https://github.com/microsoft/graphrag/) 库。特别感谢原项目的贡献者。

---

如果有任何需要进一步调整的地方，请告诉我！