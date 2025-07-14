# At the top of your module
from operator import index
import faiss
import pandas as pd
import duckdb
from tqdm import tqdm

GLOBAL_SPACE = {
}

def is_initialized(repos='default'):
    return GLOBAL_SPACE.get(repos, {}).get('is_initialized', False)

def init_repository(repos='default'):
    GLOBAL_SPACE[repos] = {
            'corpus_concepts': None,
            'model': None,
            'faiss_index': None,
            'is_initialized': False
        }

def delete_repository(repos='default'):
    if repos in GLOBAL_SPACE:
        del GLOBAL_SPACE[repos]

def build_index(model, corpus_concepts, corpus_embeddings=None, nlist=100, repos='default'):
    """
    Args:
        model: SentenceTransformer model to use for embeddings
        corpus_concepts: A dataframe with ['concept_id', 'concept_name'] columns
    """
    init_repository(repos)
    corpus_concepts = corpus_concepts[['concept_id', 'concept_name']].reset_index(drop=True)
    corpus_concepts['index'] = corpus_concepts.index
    
    
    GLOBAL_SPACE[repos]['model'] = model
    GLOBAL_SPACE[repos]['corpus_concepts'] = corpus_concepts

    if corpus_embeddings is None:
        concept_names = corpus_concepts['concept_name'].tolist()
        corpus_embeddings = model.encode(concept_names, normalize_embeddings=True)

    # Your existing code
    dimension = corpus_embeddings.shape[1]
    # faiss_index = faiss.IndexFlatIP(dimension)
    # faiss_index.add(corpus_embeddings.astype('float32'))
    
    # quantizer = faiss.IndexFlatIP(dimension)  # the other index
    # faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    # faiss_index.train(corpus_embeddings.astype('float32'))
    # faiss_index.add(corpus_embeddings.astype('float32'))
    # faiss_index = faiss.IndexHNSWFlat(dimension, M=32, metric=faiss.METRIC_INNER_PRODUCT)
    # faiss_index.hnsw.efConstruction = 200  # default is 40
    # faiss_index.hnsw.efSearch = 50         # default is 16
    # faiss_index.add(corpus_embeddings.astype('float32')) 
    
    
    faiss_index = faiss.IndexHNSWFlat(dimension, faiss.ScalarQuantizer.QT_fp16,faiss.METRIC_INNER_PRODUCT)
    faiss_index.add(corpus_embeddings.astype('float16')) 
    
    
    GLOBAL_SPACE[repos]['is_initialized'] = True
    GLOBAL_SPACE[repos]['faiss_index'] = faiss_index
    
    return corpus_embeddings
    

def search_similar(query_ids, query_texts, query_embeddings=None, top_k=5, repos='default'):
    """
    Args:
        query_texts: List of query texts to search for
        top_k: Number of top results to return
    Returns:
        A dataframe with ['query_concept_id', 'query_text', 'concept_id', 'concept_name', 'score'] columns
    """
    if not is_initialized(repos):
        raise ValueError("Index not built. Call build_index() first.")

    model = GLOBAL_SPACE[repos]['model']
    faiss_index = GLOBAL_SPACE[repos]['faiss_index']

    if query_embeddings is None:
        query_embeddings = model.encode(query_texts, normalize_embeddings=True)
    
    embeddings = query_embeddings.astype('float32')
    # faiss_index.nprobe = nprobe 
    batch_size = 1024
    scores, indices = [], []
    for i in tqdm(range(0, len(embeddings), batch_size), desc="Searching"):
        batch_embeddings = embeddings[i:i + batch_size]
        batch_scores, batch_indices = faiss_index.search(batch_embeddings, top_k)
        scores.extend(batch_scores.tolist())
        indices.extend(batch_indices.tolist())
        
    # scores, indices = faiss_index.search(query_embeddings.astype('float32'), top_k)

    # turn indices into concept_id
    corpus_concepts = GLOBAL_SPACE[repos]['corpus_concepts']

    search_results = pd.DataFrame({
        'query_concept_id': query_ids,
        'query_text': query_texts,
        'top_k_indices': indices,
        'score': scores
    }).explode(['top_k_indices', 'score']).reset_index(drop=True)
    

    search_results = duckdb.query("""
        SELECT query_concept_id, query_text, concept_id, concept_name, score
        FROM search_results
        JOIN corpus_concepts ON search_results.top_k_indices = corpus_concepts.index
        where REPLACE(LOWER(query_text), ' ', '') != REPLACE(LOWER(concept_name), ' ', '')
    """).df()
    
    return search_results

