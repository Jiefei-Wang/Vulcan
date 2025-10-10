import faiss
import pandas as pd
import duckdb
from tqdm import tqdm
import numpy as np
from modules.ModelFunctions import encode_concepts

GLOBAL_SPACE = {
}

def is_initialized(repos='default'):
    return GLOBAL_SPACE.get(repos, {}).get('is_initialized', False)

def init_repository(repos='default'):
    GLOBAL_SPACE[repos] = {
            'corpus_df': None,
            'model': None,
            'faiss_index': None,
            'is_initialized': False
        }

def delete_repository(repos='default'):
    if repos in GLOBAL_SPACE:
        del GLOBAL_SPACE[repos]

def build_index(model, corpus_ids, corpus_names, corpus_embeddings=None, repos='default'):
    """
    Args:
        model: SentenceTransformer model to use for embeddings
        corpus_ids: List of concept IDs in the corpus
        corpus_names: List of concepts for the corpus
    """
    if len(corpus_ids) != len(corpus_names):
        raise ValueError("Length of corpus_ids and corpus_names must match")
    if len(corpus_ids) == 0:
        raise ValueError("corpus_ids cannot be empty")
    
    init_repository(repos)
    corpus_df = pd.DataFrame({
        'corpus_id': corpus_ids,
        'corpus_name': corpus_names
    })
    corpus_df['index'] = corpus_df.index
    
    
    GLOBAL_SPACE[repos]['model'] = model
    GLOBAL_SPACE[repos]['corpus_df'] = corpus_df

    if corpus_embeddings is None:
        concept_names = corpus_df['corpus_name'].tolist()
        corpus_embeddings = model.encode(concept_names, normalize_embeddings=True)

    # if corpus_embeddings is a pandas DataFrame, convert to numpy array
    corpus_embeddings = np.array(list(corpus_embeddings))
    
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
    

def search_similar(query_ids, query_names, query_embeddings=None, top_k=5, repos='default'):
    """
    Args:
        query_ids: List of concept IDs to search for
        query_names: List of query names to search for
        top_k: Number of top results to return
    Returns:
        A dataframe with ['query_id', 'query_name', 'corpus_id', 'corpus_name', 'score'] columns
    """
    if not is_initialized(repos):
        raise ValueError("Index not built. Call build_index() first.")

    query_ids = list(query_ids)
    query_names = list(query_names)
    model = GLOBAL_SPACE[repos]['model']
    faiss_index = GLOBAL_SPACE[repos]['faiss_index']

    if query_embeddings is None:
        query_embeddings = encode_concepts(model, query_names)
    
    embeddings = query_embeddings.astype('float16')
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
    corpus_df = GLOBAL_SPACE[repos]['corpus_df']

    search_results = pd.DataFrame({
        'query_id': query_ids,
        'query_name': query_names,
        'top_k_indices': indices,
        'score': scores
    }).explode(['top_k_indices', 'score']).reset_index(drop=True)
    

    search_results = duckdb.query("""
        SELECT query_id, query_name, corpus_id, corpus_name, score
        FROM search_results
        JOIN corpus_df ON search_results.top_k_indices = corpus_df.index
    """).df()
    
    return search_results
