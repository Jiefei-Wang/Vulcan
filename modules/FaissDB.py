# At the top of your module
import faiss
import pandas as pd
import duckdb

GLOBAL_SPACE = {
    'corpus_concepts': None,
    'model': None,
    'is_initialized': False,
    'faiss_index': None
}

def is_initialized():
    return GLOBAL_SPACE['is_initialized']


def build_index(model, corpus_concepts):
    """
    Args:
        model: SentenceTransformer model to use for embeddings
        corpus_concepts: A dataframe with ['concept_id', 'concept_name'] columns
    """
    corpus_concepts = corpus_concepts[['concept_id', 'concept_name']].reset_index(drop=True)
    corpus_concepts['index'] = corpus_concepts.index
    
    
    GLOBAL_SPACE['model'] = model
    GLOBAL_SPACE['corpus_concepts'] = corpus_concepts
    
    concept_names = corpus_concepts['concept_name'].tolist()
    corpus_embeddings = model.encode(concept_names, normalize_embeddings=True)

    # Your existing code
    dimension = corpus_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(corpus_embeddings.astype('float32'))
    GLOBAL_SPACE['is_initialized'] = True
    GLOBAL_SPACE['faiss_index'] = faiss_index
    

def search_similar(query_ids, query_texts, top_k=5):
    """
    Args:
        query_texts: List of query texts to search for
        top_k: Number of top results to return
    Returns:
        A dataframe with ['query_concept_id', 'query_text', 'concept_id', 'concept_name', 'score'] columns
    """
    if not GLOBAL_SPACE['is_initialized']:
        raise ValueError("Index not built. Call build_index() first.")
    
    model = GLOBAL_SPACE['model']
    faiss_index = GLOBAL_SPACE['faiss_index']
    
    query_embedding = model.encode(query_texts, normalize_embeddings=True)
    scores, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
    
    # turn indices into concept_id
    corpus_concepts = GLOBAL_SPACE['corpus_concepts']

    search_results = pd.DataFrame({
        'query_concept_id': query_ids,
        'query_text': query_texts,
        'top_k_indices': indices.tolist(),
        'score': scores.tolist()
    }).explode(['top_k_indices', 'score']).reset_index(drop=True)
    

    search_results = duckdb.query("""
        SELECT query_concept_id, query_text, concept_id, concept_name, score
        FROM search_results
        JOIN corpus_concepts ON search_results.top_k_indices = corpus_concepts.index
    """).df()
    
    return search_results
