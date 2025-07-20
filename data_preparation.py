from datasets import load_dataset
from langchain.schema import Document as LangChainDocument

def load_data():
    dataset = load_dataset("rahular/simple-wikipedia")
    return dataset

def create_documents(dataset):
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
    return langchain_documents    


if __name__ == "__main__":
    dataset = load_data()
    documents = create_documents(dataset)
    print(f"Loaded {len(documents)} LangChain documents.")    