from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from fastembed import TextEmbedding
from collections import OrderedDict
import numpy as np
import hashlib
import text_clean
import model

MAX_CACHE_SIZE = 10
DOCUMENT_CACHE = OrderedDict()
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = TextEmbedding()
    return embedding_model
    
def get_text_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

def get_document_bundle(text: str):
    cleaned_text = text_clean.clean(text)
    text_hash = get_text_hash(cleaned_text)
    if text_hash in DOCUMENT_CACHE:
        DOCUMENT_CACHE.move_to_end(text_hash)
        print("CACHE HIT")
        return DOCUMENT_CACHE[text_hash]
    print("NEW DOCUMENT PROCESSING")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n", " ", ". ", "."]
    )
    chunks = splitter.split_text(cleaned_text)
   
    if chunks:
        model_embed = get_embedding_model()
        chunk_embeddings = np.asarray(
            list(model_embed.embed(chunks)),
            dtype=np.float32
        )

    else:
        chunk_embeddings = np.empty(
            (0, 0),
            dtype=np.float32
        )

    bundle = {
        "text_hash": text_hash,
        "cleaned_text": cleaned_text,
        "chunks": chunks,
        "chunk_embeddings": chunk_embeddings,
    }

    DOCUMENT_CACHE[text_hash] = bundle
    DOCUMENT_CACHE.move_to_end(text_hash)
    
    if len(DOCUMENT_CACHE) > MAX_CACHE_SIZE:
        removed_key, _ = DOCUMENT_CACHE.popitem(last=False)
        print(f"REMOVED CACHE: {removed_key}")
    return bundle

def pipeline(query, text):
    bundle = get_document_bundle(text)
    chunks = bundle["chunks"]
    chunk_embeddings = bundle["chunk_embeddings"]
    if not chunks:
        response = model.generate_answer(
            query,
            [],
            []
        )
        return response, []

    model_embed = get_embedding_model()
    query_embedding = np.asarray(
        list(model_embed.embed([query])),
        dtype=np.float32
    )

    similarity = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )[0]

    top_indx = np.argsort(similarity)[::-1][:3]

    top_chunks = [
        chunks[i]
        for i in top_indx
    ]

    response = model.generate_answer(
        query,
        top_chunks,
        similarity
    )

    return response, top_chunks
