from sentence_transformers import SentenceTransformer

def get_embeddings(query, model_emb='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_emb)
    embedding = model.encode(query).tolist()
    return embedding

def create_embeddings(text, model_emb='sentence-transformers/all-MiniLM-L6-v2'):   
    model = SentenceTransformer(model_emb)
    embedding = model.encode(text)
    return embedding

if __name__ == "__main__":
    x = create_embeddings("hello")
    print(x)
