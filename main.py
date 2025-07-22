from data_preparation import load_data, create_documents
from embeddings_and_db import get_embedding_function, create_embeddings, get_vector_store, add_embeddings_to_database
from llm_generation import get_llm, get_retriever, get_prompt_template, get_rag_chain, get_answer
import gradio as gr
import torch


def detect_device():
    print("Detecting device...")
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return "cuda", 0
    else:
        print("CUDA not available. Using CPU.")
        return "cpu", -1

def main():
    device, pipeline_device = detect_device()

    dataset = load_data()
    documents = create_documents(dataset)

    embedding_function = get_embedding_function(device)
    #embeddings = create_embeddings(documents, embedding_function)

    vector_store = get_vector_store(embedding_function, "./chroma2_wikipedia_db")
    add_embeddings_to_database(documents, vector_store)
    
    llm = get_llm(pipeline_device, model_id = "google/flan-t5-large")
    retriever = get_retriever(vector_store, k=7)

    prompt = get_prompt_template()
    rag_chain = get_rag_chain(llm,retriever,prompt)

    def answer_question(query):
        result = get_answer(rag_chain, query)
        print(f"Generated answer:\n{result['result']}\n")
        return result["result"]
    
    #query = "How does the water cycle work in nature?"
    #answer = get_answer(rag_chain,query)
    #print(answer["result"])

    demo = gr.Interface(fn=answer_question, inputs="text", outputs="text",allow_flagging="never")
    demo.launch()
    print("Launching Gradio interface...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during execution: {e}")   