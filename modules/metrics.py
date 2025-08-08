import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from collections import defaultdict
from modules.ModelFunctions import encode_concepts

def evaluate_embedding_similarity_with_mrr(model, data, threshold=0.8):
    """
    Evaluate sentence embedding similarity using dot product similarity and return metrics including MRR. 
    query_name and query_id are the key concept to evaluate

    Parameters:
    - model: A sentence-transformers model with `.encode()` method.
    - data: A DataFrame with ['corpus_name', 'query_name', 'corpus_id', 'query_id', 'label'] columns.
    - threshold: Threshold for binarizing similarity

    Returns:
    - Dictionary of evaluation metrics including MRR
    """

    # Step 1: Encode sentences
    corpus_emb = encode_concepts(model, data['corpus_name'].tolist(), normalize_embeddings=True)
    query_emb = encode_concepts(model, data['query_name'].tolist(), normalize_embeddings=True)

    # Step 2: Compute similarity
    similarities = np.sum(corpus_emb * query_emb, axis=1)
    labels = data['label'].values
    metrics = evaluate_performance(data['query_id'].tolist(), similarities, labels, threshold)
    
    return metrics





def evaluate_performance(query_ids, similarities, labels, threshold=0.8):
    """
    Evaluate sentence embedding similarity using dot product similarity and return metrics including MRR. 
    query_name and query_id are the key concept to evaluate

    Parameters:
    - query_ids: List of query IDs.
    - similarities: Array-like of similarity scores.
    - labels: Array-like of binary labels (0 or 1).
    - threshold: Threshold for binarizing similarity

    Returns:
    - Dictionary of evaluation metrics including MRR
    """
    query_ids = np.array(query_ids)
    similarities = np.array(similarities)
    labels = np.array(labels)
    preds = (similarities >= threshold).astype(int)

    metrics = {
        'roc_auc': roc_auc_score(labels, similarities),
        'average_precision': average_precision_score(labels, similarities),
        'f1_score': f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'accuracy': np.mean(preds == labels)
    }

    # Step 4: Compute MRR
    df = pd.DataFrame({
        'query_id': query_ids,
        'similarity': similarities,
        'label': labels
    })

    # Group by query_id, rank all candidates by similarity
    grouped = df.groupby('query_id')
    index_based_metrics = []

    for _, group in grouped:
        sorted_group = group.sort_values('similarity', ascending=False).reset_index(drop=True)
        positive_indices = sorted_group.index[sorted_group['label'] == 1].tolist()
        if positive_indices:
            best = index_to_metric(positive_indices[0])
            worst = index_to_metric(positive_indices[-1])
        else:
            best = worst = index_to_metric(None)
            
        best = {f'best_{k}': v for k, v in best.items()}
        worst = {f'worst_{k}': v for k, v in worst.items()}
        index_based_metrics.append(best | worst)
            
    # to data.frame
    index_based_metrics = pd.DataFrame(index_based_metrics)
    index_based_metrics[['best_reciprocal_rank', 'worst_reciprocal_rank']] 
    # column means
    index_based_metrics = index_based_metrics.mean().to_dict()
    
    metrics = metrics | index_based_metrics
    return metrics


def index_to_metric(index):
    if index is not None:
        hit1 = index < 1
        hit3 = index < 3
        hit5 = index < 5
        hit10 = index < 10
        hit20 = index < 20
        hit50 = index < 50
        hit100 = index < 100
        reciprocal_ranks = 1.0 / (index + 1)
    else:
        hit1 = hit3 = hit5 = hit10 = hit20 = hit50 = hit100 = reciprocal_ranks = 0.0
    return {
        'hit1': hit1,
        'hit3': hit3,
        'hit5': hit5,
        'hit10': hit10,
        'hit20': hit20,
        'hit50': hit50,
        'hit100': hit100,
        'reciprocal_rank': reciprocal_ranks
    }