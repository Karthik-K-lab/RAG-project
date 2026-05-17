from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from fastembed import TextEmbedding
import numpy as np

import text_clean
import model

embedding_model = None


def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        embedding_model = TextEmbedding()

    return embedding_model


def pipeline(query, text):
    text = text_clean.clean(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n", " ", ". ", "."]
    )

    chunks = splitter.split_text(text)

    if not chunks:
        response = model.generate_answer(query, [], np.array([]))
        return response, []

    model_embed = get_embedding_model()

    chunk_embeddings = np.array(list(model_embed.embed(chunks)))
    query_embedding = np.array(list(model_embed.embed([query])))[0].reshape(1, -1)

    similarity = cosine_similarity(query_embedding, chunk_embeddings)[0]

    top_indx = np.argsort(similarity)[::-1][:3]
    top_chunks = [chunks[i] for i in top_indx]

    response = model.generate_answer(query, top_chunks, similarity)

    return response, top_chunks
