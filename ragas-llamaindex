import os
from itertools import chain
from ragas.testset.graph import NodeType
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.transforms.extractors.llm_based import ThemesExtractor
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.utils import num_tokens_from_string
from ragas.llms import BaseRagasLLM, LlamaIndexLLMWrapper
from ragas.embeddings.base import LlamaIndexEmbeddingsWrapper
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms, default_transforms

# LlamaIndex 相关导入
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
eval_llm = DeepSeek(model="deepseek-chat", api_key="**************************")
emb_api_key = "*******************************"
eval_embedding_model = SiliconFlowEmbedding(
    model="BAAI/bge-m3",  # 根据需要选择模型
    api_key=emb_api_key,
    base_url="https://api.siliconflow.cn/v1/embeddings",
    encoding_format="float"
)

llm = eval_llm
embeddings = eval_embedding_model

# Helper class to adapt LlamaIndex Document for Ragas
class RagasDocument:
    def __init__(self, text, metadata):
        self.page_content = text
        self.metadata = metadata

def filter_doc_with_num_tokens(node, min_num_tokens=500):
    return (
        node.type == NodeType.DOCUMENT
        and num_tokens_from_string(node.properties["page_content"]) > min_num_tokens
    )

def filter_chunks(node):
    return node.type == NodeType.CHUNK

def split(filename):
    print(f'Processing {filename}')
    with open(filename) as f:
        doc_text = f.read()
        # 使用 LlamaIndex 的文档处理方式
        doc = Document(text=doc_text)
        # 使用 LlamaIndex 的分割器
        text_splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
        nodes = text_splitter.get_nodes_from_documents([doc])
        # 转换为与原代码兼容的格式
        splits = [Document(text=node.text, metadata=node.metadata) for node in nodes]
    return splits

# 检查 docs 目录是否存在，如果不存在则使用 files 目录
docs_dir = 'docs' if os.path.exists('docs') else 'files'
# 看上去这里应该以document为单位读进来。
docsplits = list(chain(*[split(f'{docs_dir}/{filename}') for filename in os.listdir(docs_dir)]))

personas = [
    Persona(
        name="curious engineer",
        role_description="An development engineer who wants to know the different types of AI projects",
    ),
]

theme_extractor = ThemesExtractor(
    llm=LlamaIndexLLMWrapper(llm), filter_nodes=lambda node: filter_chunks(node)
)

# Adapt LlamaIndex documents for Ragas default_transforms
adapted_docsplits = [RagasDocument(doc.text, doc.metadata) for doc in docsplits]

# 使用 LlamaIndex 的包装器
# 这是一个提前包装好的pipeline，里面包含了计算主题，计算embedding，计算主题相似度，计算ner，计算ner相似度
# 然后可以利用这些算出来的东西构建一个知识图谱。
transforms = default_transforms(
    documents=adapted_docsplits, # Use adapted documents
    llm=LlamaIndexLLMWrapper(llm),
    embedding_model=LlamaIndexEmbeddingsWrapper(embeddings),
)

# 创建节点
# 应该不用NodeType.CHUNK,而是使用NodeType.DOCUMENT, 否则HeadlinesExtractor，HeadlineSplitter，SummaryExtractor不work。
# 详情参考default_transforms的说明文档， 然后前面感觉应该不能直接chunk，而是要以document为单位读进来才对
nodes = [Node(type=NodeType.CHUNK, 
              properties={'page_content': docs.text, 
                          'document_metadata': docs.metadata})
        for docs in docsplits]
graph = KnowledgeGraph(nodes=nodes)
apply_transforms(graph, transforms)

# 使用 LlamaIndex 版本的 TestsetGenerator
generator = TestsetGenerator.from_llama_index(
    llm=llm, embedding_model=embeddings, 
    knowledge_graph=graph)
generator.persona_list = personas
dataset = generator.generate(testset_size=3)

df = dataset.to_pandas()
df.to_csv('testset.csv', index=False)
