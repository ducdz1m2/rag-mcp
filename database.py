from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache
import os

# Get absolute path to faiss_index
VECTOR_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

def load_db():
    embeddings = get_embeddings()
    # Load từ local lên, cho phép giải nén an toàn
    return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)