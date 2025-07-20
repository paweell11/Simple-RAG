from data_preparation import load_data, create_documents
from embeddings_and_db import get_embedding_function, create_embeddings, get_vector_store, add_embeddings_to_database
from llm_generation import get_llm, get_retriever, get_prompt_template, get_rag_chain, get_answer


def main():
    dataset = load_data()
    print("load dataset")
    documents = create_documents(dataset)
    print("cerate dataset")
    embedding_function = get_embedding_function("cuda")
    embeddings = create_embeddings(documents, embedding_function)
    print("cerate embdedings")
    vector_store = get_vector_store(embedding_function, "./chroma2_wikipedia_db")
    add_embeddings_to_database(documents, vector_store)
    print("embed in database")
    llm = get_llm(0, model_id = "google/flan-t5-large")
    retriever = get_retriever(vector_store, k=7)
    prompt = get_prompt_template()
    rag_chain = get_rag_chain(llm,retriever,prompt)
    query = "How does the water cycle work in nature?"
    answer = get_answer(rag_chain,query)
    print(answer["result"])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error during execution: {e}")   