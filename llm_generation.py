from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def get_llm(pipeline_device, model_id = "google/flan-t5-large", ):
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
    return vector_store.as_retriever(search_kwargs={"k": k})

def get_prompt_template():
    return PromptTemplate(
        input_variables=["context", "question"],
        template = """
You are a helpful assistant. Using only the information below, write a detailed and informative answer to the question.

{context}

Question: {question}
Answer:
"""
    )

def get_rag_chain(llm,retriever,prompt):
    return RetrievalQA.from_chain_type(llm=llm,retriever=retriever,chain_type_kwargs={"prompt": prompt},return_source_documents=True,verbose=True)    

def get_answer(rag_chain,query):
    return rag_chain(query)
