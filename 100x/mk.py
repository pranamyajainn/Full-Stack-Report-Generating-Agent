from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# global default
Settings.embed_model = OpenAIEmbedding()

documents = SimpleDirectoryReader("/Users/pranamyajain/PycharmProjects/FullStackReportGeneratingAgent/100x").load_data()

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("analyse the data file and tell me what is in it?")
print(response)