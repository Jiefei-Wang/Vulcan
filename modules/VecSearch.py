import pandas as pd
import duckdb
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

GLOBAL_SPACE = {
}

def get_device():
    """Get the best available device (GPU if available, otherwise CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def print_gpu_info():
    """Print GPU information for debugging"""
    device = get_device()
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def is_initialized(repos='default'):
    return GLOBAL_SPACE.get(repos, {}).get('is_initialized', False)

def init_repository(repos='default'):
    GLOBAL_SPACE[repos] = {
            'corpus_df': None,
            'model': None,
            'corpus_embeddings_gpu': None,
            'device': get_device(),
            'is_initialized': False
        }

def delete_repository(repos='default'):
    if repos in GLOBAL_SPACE:
        del GLOBAL_SPACE[repos]

def build_index(model, corpus_ids, corpus_names, corpus_embeddings=None, repos='default'):
    """
    GPU-accelerated index building for fast similarity search.
    
    Args:
        model: SentenceTransformer model to use for embeddings
        corpus_ids: List or Series of concept IDs in the corpus
        corpus_names: List or Series of concepts for the corpus
        corpus_embeddings: Optional pre-computed embeddings
        repos: Repository name for multiple indexes
    """
    if len(corpus_ids) != len(corpus_names):
        raise ValueError("Length of corpus_ids and corpus_names must match")
    if len(corpus_ids) == 0:
        raise ValueError("corpus_ids cannot be empty")
    
    init_repository(repos)
    device = GLOBAL_SPACE[repos]['device']
    
    print_gpu_info()
    
    # Create corpus DataFrame
    corpus_df = pd.DataFrame({
        'corpus_id': corpus_ids,
        'corpus_name': corpus_names
    })
    corpus_df['index'] = corpus_df.index
    
    GLOBAL_SPACE[repos]['model'] = model
    GLOBAL_SPACE[repos]['corpus_df'] = corpus_df
    
    # Generate embeddings if not provided
    if corpus_embeddings is None:
        concept_names = corpus_df['corpus_name'].tolist()
        print(f"Encoding {len(concept_names)} concepts")
        corpus_embeddings = model.encode(concept_names, normalize_embeddings=True, show_progress_bar=True)
    
    # Handle different input types
    if isinstance(corpus_embeddings, pd.Series):
        corpus_embeddings = np.array(corpus_embeddings.tolist())
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = np.array(corpus_embeddings)
    
    # Convert to tensor and move to GPU
    corpus_embeddings_tensor = torch.tensor(corpus_embeddings, dtype=torch.float32)
    corpus_embeddings_gpu = corpus_embeddings_tensor.to(device)
    
    print(f"Moved embeddings to {device}")
    print(f"Corpus embeddings shape: {corpus_embeddings_gpu.shape}")
    print(f"GPU memory used: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB" if device.type == 'cuda' else "")
    
    GLOBAL_SPACE[repos]['corpus_embeddings_gpu'] = corpus_embeddings_gpu
    GLOBAL_SPACE[repos]['is_initialized'] = True
    
    return corpus_embeddings


def search_similar(query_ids, query_names, query_embeddings=None, top_k=5, repos='default'):
    """
    GPU-accelerated similarity search using PyTorch matrix operations.
    
    Args:
        query_ids: List or Series of concept IDs to search for
        query_names: List or Series of query names to search for
        query_embeddings: Optional list of precomputed embeddings for the queries
        top_k: Number of top results to return (default 5, can handle up to 50 efficiently)
        repos: Repository name to search in
    Returns:
        A dataframe with ['query_id', 'query_name', 'corpus_id', 'corpus_name', 'score'] columns
    """
    if not is_initialized(repos):
        raise ValueError("Index not built. Call build_index() first.")
    
    model = GLOBAL_SPACE[repos]['model']
    corpus_embeddings_gpu = GLOBAL_SPACE[repos]['corpus_embeddings_gpu']
    corpus_df = GLOBAL_SPACE[repos]['corpus_df']
    device = GLOBAL_SPACE[repos]['device']
    
    # Generate query embeddings if not provided
    if query_embeddings is None:
        print(f"Encoding {len(query_names)} queries...")
        query_embeddings = model.encode(query_names, normalize_embeddings=True)
    
    # Handle different input types and convert to tensor
    if isinstance(query_embeddings, pd.Series):
        query_embeddings = np.array(query_embeddings.tolist())
    elif isinstance(query_embeddings, list):
        query_embeddings = np.array(query_embeddings)
    
    query_embeddings_tensor = torch.tensor(query_embeddings, dtype=torch.float32)
    query_embeddings_gpu = query_embeddings_tensor.to(device)
    
    print(f"Query embeddings shape: {query_embeddings_gpu.shape}")
    print(f"Corpus embeddings shape: {corpus_embeddings_gpu.shape}")
    
    # GPU-accelerated similarity computation using matrix multiplication
    batch_size = 1024  # Process queries in batches to manage memory
    all_scores = []
    all_indices = []
    
    num_queries = query_embeddings_gpu.shape[0]
    
    for i in tqdm(range(0, num_queries, batch_size), desc="GPU Similarity Search"):
        end_idx = min(i + batch_size, num_queries)
        query_batch = query_embeddings_gpu[i:end_idx]
        
        # Compute cosine similarity: query_batch @ corpus_embeddings.T
        # Both are already normalized, so dot product = cosine similarity
        similarity_scores = torch.mm(query_batch, corpus_embeddings_gpu.T)
        
        # Get top-k results
        top_scores, top_indices = torch.topk(similarity_scores, k=top_k, dim=1)
        
        # Move results back to CPU for processing
        all_scores.extend(top_scores.cpu().numpy().tolist())
        all_indices.extend(top_indices.cpu().numpy().tolist())
    
    # Create results DataFrame
    search_results = pd.DataFrame({
        'query_id': query_ids,
        'query_name': query_names,
        'top_k_indices': all_indices,
        'score': all_scores
    }).explode(['top_k_indices', 'score']).reset_index(drop=True)
    
    # Join with corpus data to get concept names and filter exact matches
    search_results = duckdb.query("""
        SELECT query_id, query_name, corpus_id, corpus_name, score
        FROM search_results
        JOIN corpus_df ON search_results.top_k_indices = corpus_df.index
        WHERE REPLACE(LOWER(query_name), ' ', '') != REPLACE(LOWER(corpus_name), ' ', '')
    """).df()
    
    print(f"GPU Search completed: {len(search_results)} results returned")
    if device.type == 'cuda':
        print(f"Peak GPU memory used: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
        torch.cuda.reset_peak_memory_stats(device)
    
    return search_results

