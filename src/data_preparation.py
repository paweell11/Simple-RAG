from datasets import load_dataset
from langchain.schema import Document as LangChainDocument

def load_data():
    print("Loading dataset...")
    dataset = load_dataset("rahular/simple-wikipedia")
    return dataset

def create_documents(dataset):
    print("Creating LangChain documents...")
    langchain_documents = []
    for i, record in enumerate(dataset["train"]):
        text_content = record["text"]
            
        if not text_content.strip():
            continue
            
        metadata = {
            "source": "simple_wikipedia",
            "original_index": i,
            "text_length": len(text_content),
        }
            
        doc = LangChainDocument(
            page_content=text_content,
            metadata=metadata
        )
            
        langchain_documents.append(doc)       
    print(f"Created {len(langchain_documents)} LangChain documents.")
    return langchain_documents    
