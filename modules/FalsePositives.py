import duckdb
from modules.FaissDB import build_index, search_similar, is_initialized


def get_false_positives(model, 
                        corpus_concepts, 
                        query_concepts = None, 
                        blacklist=None,
                        n_fp=50, repos='default'):
    """
    Args:
        model: SentenceTransformer model to use for embeddings
        corpus_concepts: A dataframe with ['concept_id', 'concept_name'] columns
        query_concepts: A dataframe with ['concept_id', 'concept_name'] columns, if None, use corpus_concepts
        blacklist: A dataframe with ['concept_id1', 'concept_id2'] columns, pairs to exclude from the results, if None, no pairs are excluded
        n_fp: Number of false positive pairs to generate
    """
    
    target_embeddings = None
    if not is_initialized(repos=repos):
        target_embeddings = build_index(model, corpus_concepts, repos=repos)
    if query_concepts is None:
        query_concepts = corpus_concepts
        if target_embeddings is not None:
            query_embeddings = target_embeddings
        else:
            query_embeddings = None
    else:
        query_embeddings = None
        
    fp = search_similar(
        query_ids=query_concepts['concept_id'].to_list(),
        query_texts=query_concepts['concept_name'].to_list(),
        query_embeddings=query_embeddings,
        repos=repos,
        top_k=n_fp)
    
    fp = fp.rename(columns={
        'query_concept_id': 'concept_id1',
        'query_text': 'sentence1',
        'concept_id': 'concept_id2',
        'concept_name': 'sentence2'
    })
    fp = fp[['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'score']]
    
    # Exclude pairs in the blacklist
    if blacklist is not None:
        fp = duckdb.query("""
        SELECT sentence1, sentence2, concept_id1, concept_id2, score
        FROM fp
        ANTI JOIN blacklist
        ON fp.concept_id1 = blacklist.concept_id1
        AND fp.concept_id2 = blacklist.concept_id2
        """).df()
    
    
    fp['label'] = 0
    
    return fp