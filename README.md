# RAG Web Application

This project is a lightweight Retrieval-Augmented Generation (RAG) web application built using React, FastAPI, FastEmbed, and Groq APIs. The main idea of the project is to allow users to provide any textual content and ask questions related to that content. Instead of directly generating answers like a normal chatbot, the system first retrieves the most relevant parts of the content and then generates an answer based on that context.

The project was mainly focused on building a lightweight and efficient semantic question-answering system that can run smoothly on cloud platforms without requiring heavy infrastructure.

Here is the working link of the project, https://rag-project-khaki.vercel.app hope you enjoy it. 

---

# Problem Statement

Most traditional chatbots and AI systems generate responses without properly understanding the provided content, which can lead to hallucinated or inaccurate answers. Also, repeatedly training of the model on new data increases response time and memory usage.

The goal of this project was to build a system that:
- Retrieves relevant information before generating answers
- Improves answer accuracy
- Works efficiently in lightweight cloud environments

---

# Solution

To solve this problem, a Retrieval-Augmented Generation (RAG) pipeline was implemented.

The content provided by the user is first cleaned and split into smaller chunks. These chunks are then converted into embeddings using FastEmbed. When the user asks a question, semantic similarity is calculated between the question and the stored chunks using cosine similarity. The most relevant chunks are selected and sent along with the question to the Groq LLM API for generating a context-aware response.

To improve performance, caching and lazy loading techniques were also implemented so that repeated content does not get processed again and again.

---

# Technologies Used

## Frontend
- React
- Vite
- JavaScript
- CSS

## Backend
- FastAPI
- Python
- Pydantic

## AI / RAG Components
- FastEmbed
- scikit-learn
- LangChain Text Splitters
- Groq API
- Llama 3.1 8B Instant

## Deployment
- Vercel (Frontend)
- Render (Backend)

---

# Project Flow

```text
User Input
   ↓
React Frontend
   ↓
FastAPI Backend
   ↓
Text Cleaning & Chunking
   ↓
FastEmbed Embeddings
   ↓
Cosine Similarity Search
   ↓
Relevant Chunk Retrieval
   ↓
Groq LLM Response Generation
   ↓
Answer Returned to User
```

---

# Features

- Semantic question answering
- Context-aware response generation
- Lightweight RAG architecture
- LRU caching for optimization
- Lazy loading for reduced memory usage
- Cloud deployment using Vercel and Render


---

# Future Improvements

Some future improvements planned for the project include:
- PDF/document uploads
- Multi-document support
- Persistent vector database integration
- Chat history support
- Streaming responses
