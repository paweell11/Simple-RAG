import os
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_llm(pipeline_device, use_gemini=True):
    if use_gemini:
        print("Initializing model via API: Gemini 3 Flash...")
        
        load_dotenv() 
        
        return ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0.1,
            max_retries=2
        )
        
    else:
        print("Initializing local model: flan-t5-large (this might take a while)...")
        
        model_id = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=pipeline_device, 
            do_sample=True,
            max_new_tokens=200,
            temperature=0.7,
            repetition_penalty=1.0,
        )
        return HuggingFacePipeline(pipeline=pipe)


def get_retriever(vector_store, k=7):
    print(f"Creating retriever with top-{k} search...")
    return vector_store.as_retriever(search_kwargs={"k": k})


def get_prompt_template():
    print("Creating prompt template...")
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Using only the information below, write a detailed and informative answer to the question.

{context}

Question: {question}
Answer:
""",
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain(llm, retriever, prompt):
    print("Building RAG chain...")
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def get_answer(rag_chain, query):
    print(f"Answering query: {query}")
    return rag_chain.invoke(query)
