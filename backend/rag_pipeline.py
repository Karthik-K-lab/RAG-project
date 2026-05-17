      
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sklearn.metrics.pairwise import cosine_similarity
from fastembed import TextEmbedding
import numpy as np
import pprint
import text_clean
import model

def pipeline(query, text):

    text = text_clean.clean(text)


    splitter = RecursiveCharacterTextSplitter(chunk_size = 200,
                                            chunk_overlap = 50,
                                            separators = ["\n"," ", ". ","."])

    chunks = splitter.split_text(text)

    embedding_model = TextEmbedding()
    embedded_data = list(embedding_model.embed(chunks))

    embedded_query = list(embedding_model.embed(query))

    similarity = cosine_similarity(embedded_query, embedded_data)[0]
    top_indx = np.argsort(similarity)[::-1][:3]
    top_chunks  = [chunks[i] for i  in top_indx]


    

    response = model.generate_answer(query, top_chunks, similarity)

    print("Query: ", query)
    print("Answer: ", response)
    print("Top Chunks are:\n")
    for idx, chunk in enumerate(top_chunks):
        print(idx+1,":",chunk)
    print("---------------------------\n")
    
    return response, top_chunks