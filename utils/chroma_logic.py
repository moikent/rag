from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
import ast
import os
import chromadb
from chromadb.utils import embedding_functions
# default_ef = embedding_functions.DefaultEmbeddingFunction()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

PERSISTENT_PATH = './db/chroma.db'
# client = chromadb.PersistentClient(path="./db/chroma.db")

# #Global variables
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# pinecone = Pinecone(api_key=PINECONE_API_KEY)

# delete index
def delete_pinecone_index(index_name):
    print(f"Deleting index '{index_name}' if it exists.")
    try:
        pinecone.delete_index(index_name)
        print(f"Index '{index_name}' successfully deleted.")
    except Exception as e:
        print(f"index '{index_name}' not found no action taken.")


# create index if needed
def get_pinecone_index(index_name):
    chroma_client = chromadb.PersistentClient(PERSISTENT_PATH)
    collection = chroma_client.get_or_create_collection(name=index_name)
    size = collection.count()
    index_created = False if size > 0 else True
    return collection, index_created

# Function to upsert data
def upsert_data(collection, df):
    # chroma_client = chromadb.Client()
    print("Start: Upserting data to Pinecone index")
    # prepped = []

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        meta = ast.literal_eval(row['metadata'])
        # prepped.append({'id': row['id'], 
        #                 'values': row['values'],
        #                 'metadata': meta})
        # if len(prepped) >= 200: # batching upserts
        #     index.upsert(prepped)
        #     prepped = []
        collection.upsert(
            embeddings=[row['values']],
            metadatas=[meta],
            ids=[row['id']]
        )

    # # Upsert any remaining entries after the loop
    # if len(prepped) > 0:
    #     index.upsert(prepped)
    
    print("Done: Data upserted to ChromaDB")
    return collection

