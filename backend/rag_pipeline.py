from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

import text_clean
import model

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def pipeline(query, text):

    text = text_clean.clean(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n", " ", ". ", "."]
    )

    chunks = splitter.split_text(text)

    embedded_data = embedding_model.encode(chunks)

    embedded_query = embedding_model.encode([query])

    similarity = cosine_similarity(
        embedded_query,
        embedded_data
    )[0]

    top_indx = np.argsort(similarity)[::-1][:3]

    top_chunks = [chunks[i] for i in top_indx]

    response = model.generate_answer(
        query,
        top_chunks,
        similarity
    )

    return response, top_chunks
