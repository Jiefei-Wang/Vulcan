import duckdb
from modules.FaissDB import build_index, search_similar, is_initialized
import pandas as pd

def getFalsePositives(model, 
                    corpus_ids,
                    corpus_names, 
                    query_ids = None,
                    query_names = None, 
                    blacklist_from = [],
                    blacklist_to = [],
                    n_fp=50, repos='default'):
    """
    Args:
        model: SentenceTransformer model to use for embeddings
        corpus_ids: List of concept IDs in the corpus
        corpus_names: List of concepts for the corpus
        query_ids: List of concept IDs to query against the corpus
        query_names: List of concepts for the query
        blacklist_from: List of concept IDs in the corpus
        blacklist_to: List of concept IDs in the query
        pairs to exclude from the results, if None, no pairs are excluded
        n_fp: Number of false positive pairs to generate
    """
    
    query_same_as_corpus = query_ids is None and query_names is None
    if query_same_as_corpus:
        query_ids = corpus_ids
        query_names = corpus_names
    # check the length
    if len(corpus_ids) != len(corpus_names):
        raise ValueError("Length of corpus_ids and corpus_concepts must match")
    if len(query_ids) != len(query_names):
        raise ValueError("Length of query_ids and query_concepts must match")
    if len(blacklist_from) != len(blacklist_to):
        raise ValueError("Length of blacklist_from and blacklist_to must match")
    
    corpus_df = pd.DataFrame({
        'corpus_id': corpus_ids,
        'corpus_name': corpus_names
    })
    
    query_df = pd.DataFrame({
        'query_id': query_ids,
        'query_name': query_names
    })
    
    blacklist_df = pd.DataFrame({
        'corpus_id': blacklist_from,
        'query_id': blacklist_to
    })


    corpus_embeddings = None
    query_embeddings = None
    if not is_initialized(repos=repos):
        corpus_embeddings = build_index(model, corpus_ids = corpus_ids, corpus_names=corpus_names, repos=repos)
        if query_same_as_corpus:
            query_embeddings = corpus_embeddings
        
    fp = search_similar(
        query_ids=query_df['query_id'].to_list(),
        query_names=query_df['query_name'].to_list(),
        query_embeddings=query_embeddings,
        repos=repos,
        top_k=n_fp)
    
    # Exclude pairs in the blacklist
    if len(blacklist_df)>0:
        fp = duckdb.query("""
        SELECT query_id, query_name, corpus_id, corpus_name,score
        FROM fp
        ANTI JOIN blacklist_df
        ON fp.query_id = blacklist_df.query_id
        AND fp.corpus_id = blacklist_df.corpus_id
        """).df()
    
    fp['label'] = 0
    fp = fp[['query_id', 'query_name', 'corpus_id', 'corpus_name', 'score', 'label']]
    
    return fp