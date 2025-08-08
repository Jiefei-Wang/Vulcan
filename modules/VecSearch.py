import faiss
import pandas as pd
import duckdb
from tqdm import tqdm
import numpy as np

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
        corpus_ids: List or Series of concept IDs in the corpus
        corpus_names: List or Series of concepts for the corpus
    """
    
    return corpus_embeddings


def search_similar(query_ids, query_names, query_embeddings=None, top_k=5, repos='default'):
    """
    Args:
        query_ids: List or Series of concept IDs to search for
        query_names: List or Series of query names to search for
        query_embeddings: Optional list of precomputed embeddings for the queries
        top_k: Number of top results to return
    Returns:
        A dataframe with ['query_id', 'query_name', 'corpus_id', 'corpus_name', 'score'] columns
    """
    
    
    return search_results

