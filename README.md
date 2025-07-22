# 🔍 Simple-RAG

This is a basic Retrieval-Augmented Generation (RAG) project that loads data from Simple Wikipedia, stores it in a vector database, and lets you ask questions based on that data — using a simple Gradio interface.


### ⚙️ How it works:
1. It loads articles from [Simple Wikipedia](https://simple.wikipedia.org).
2. Converts the text into LangChain-compatible documents.
3. Generates embeddings using the `all-MiniLM-L6-v2` model.
4. Stores them in a vector database using ChromaDB.
5. Uses a language model (`google/flan-t5-large`) to generate answers based on retrieved context.
6. Provides a clean interface for asking questions using Gradio.


### 🧠 Want to Understand How RAG Works?

If you're curious about what's going on behind the scenes, check out the included **notebook**:  
It contains a more detailed explanation of:
- How RAG is built step-by-step
- What each component (retriever, embedding, LLM) does
- How you can modify or expand the pipeline

It's a great place to explore, experiment, and learn more about the inner workings.

### 🚀 How to Run the Project
1. **Clone the repository**
```bash
git clone https://github.com/paweell11/Simple-RAG.git
cd Simple-RAG
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Install PyTorch (not included in requirements.txt)**

For example, on most systems with CPU only, you can run:
```bash
pip install torch torchvision torchaudio
```

4. **Run the app** 
```bash
python main.py
```
or explore it interactively with:
```bash
jupyter notebook
```

