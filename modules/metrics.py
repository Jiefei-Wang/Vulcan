import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from collections import defaultdict

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
    corpus_emb = model.encode(data['corpus_name'].tolist(), normalize_embeddings=True)
    query_emb = model.encode(data['query_name'].tolist(), normalize_embeddings=True)

    # Step 2: Compute similarity
    similarities = np.sum(corpus_emb * query_emb, axis=1)
    labels = data['label'].values
    preds = (similarities >= threshold).astype(int)

    # Step 3: Compute standard metrics
    metrics = {
        'roc_auc': roc_auc_score(labels, similarities),
        'average_precision': average_precision_score(labels, similarities),
        'f1_score': f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'accuracy': np.mean(preds == labels)
    }

    # Step 4: Compute MRR
    df = data.copy()
    df['similarity'] = similarities

    # Group by query_id, rank all candidates by similarity
    grouped = df.groupby('query_id')
    reciprocal_ranks = []

    for _, group in grouped:
        sorted_group = group.sort_values('similarity', ascending=False).reset_index(drop=True)
        positive_indices = sorted_group.index[sorted_group['label'] == 1].tolist()
        if positive_indices:
            reciprocal_ranks.append(1.0 / (positive_indices[0] + 1))  # rank is 1-based
        else:
            raise ValueError("No positive labels found for concept_id1: {}".format(group['concept_id1'].iloc[0]))

    if reciprocal_ranks:
        metrics['MRR'] = np.mean(reciprocal_ranks)
    else:
        metrics['MRR'] = 0.0

    return metrics
