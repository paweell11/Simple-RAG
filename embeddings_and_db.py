from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil
import os

def get_embedding_function(device):
    print(f"Initializing embedding function on device: {device}...")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs={'device': device})

def create_embeddings(langchain_documents,embedding_function):
    print(f"Creating embeddings for {len(langchain_documents)} documents...")
    texts = [doc.page_content for doc in langchain_documents]
    embeddings = embedding_function.embed_documents(texts)
    return embeddings

def get_vector_store(embedding_function, persist_dir, reset=True):
    if reset and os.path.exists(persist_dir):
        print(f"Removing existing vector store at: {persist_dir}")
        shutil.rmtree(persist_dir)
    print(f"Creating new Chroma vector store at: {persist_dir}")
    return Chroma(collection_name="simple_wikipedia_collection",embedding_function=embedding_function,persist_directory=persist_dir,)    

def add_embeddings_to_database(langchain_documents, vector_store, BATCH_SIZE = 5000):
    print(f"Adding documents to vector store in batches of {BATCH_SIZE}...")
    for i in range(0, len(langchain_documents), BATCH_SIZE):
        batch = langchain_documents[i:i + BATCH_SIZE]
        vector_store.add_documents(batch)