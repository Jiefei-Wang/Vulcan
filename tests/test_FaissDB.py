from modules.FaissDB import is_initialized, build_index, search_similar, init_repository, delete_repository
import pandas as pd
import numpy as np
from modules.ModelFunctions import get_ST_model

# Test initialization state
assert not is_initialized(), "Repository should not be initialized initially"

# Test repository management
init_repository('test_repo')
assert not is_initialized('test_repo'), "Repository should not be initialized after init_repository"
delete_repository('test_repo')
assert not is_initialized('test_repo'), "Repository should not be initialized after deletion"

# Load model for testing
base_model = 'ClinicalBERT'
model, tokenizer = get_ST_model(base_model)

# Test build_index with valid data
corpus_ids = ['C001', 'C002', 'C003', 'C004', 'C005']
corpus_names = [
    'Diabetes mellitus',
    'Hypertension',
    'Heart failure',
    'Chronic kidney disease',
    'Pneumonia'
]

embeddings = build_index(model, corpus_ids, corpus_names, repos='test_build')
assert is_initialized('test_build'), "Repository should be initialized after build_index"
assert embeddings is not None, "build_index should return embeddings"
assert embeddings.shape[0] == len(corpus_ids), f"Expected {len(corpus_ids)} embeddings, got {embeddings.shape[0]}"

# Test build_index error cases - mismatched lengths
error_caught = False
try:
    build_index(model, ['C001'], ['name1', 'name2'], repos='test_error1')
except ValueError:
    error_caught = True
assert error_caught, "Should have raised ValueError for mismatched lengths"

# Test build_index error cases - empty corpus  
error_caught = False
try:
    build_index(model, [], [], repos='test_error2')
except ValueError:
    error_caught = True
assert error_caught, "Should have raised ValueError for empty corpus"

# Test search_similar functionality
query_ids = ['Q001', 'Q002']
query_names = ['Diabetes', 'High blood pressure']

results = search_similar(query_ids, query_names, top_k=3, repos='test_build')
assert isinstance(results, pd.DataFrame), "search_similar should return a DataFrame"
assert len(results) == len(query_ids) * 3, "Should return top_k results for each query"

expected_columns = ['query_id', 'query_name', 'corpus_id', 'corpus_name', 'score']
assert all(col in results.columns for col in expected_columns), f"Missing columns in results. Expected: {expected_columns}, Got: {list(results.columns)}"


unique_queries = results['query_id'].nunique()
assert unique_queries <= len(query_ids), f"Expected at most {len(query_ids)} unique queries, got {unique_queries}"

# Test search without initialized index
error_caught = False
try:
    search_similar(['Q001'], ['Test'], repos='non_existent_repo')
except ValueError:
    error_caught = True
assert error_caught, "Should have raised ValueError for uninitialized repository"

# Test with pre-computed embeddings
test_embeddings = np.random.rand(len(corpus_ids), 384).astype(np.float32)  # Assuming 384-dim embeddings

build_index(model, corpus_ids, corpus_names, corpus_embeddings=test_embeddings, repos='test_precomputed')
assert is_initialized('test_precomputed'), "Repository should be initialized with pre-computed embeddings"

# Test search with pre-computed query embeddings
query_embeddings = np.random.rand(len(query_ids), 384).astype(np.float32)
results = search_similar(query_ids, query_names, query_embeddings=query_embeddings, top_k=2, repos='test_precomputed')
assert len(results) <= len(query_ids) * 2, "Should return at most top_k results per query"

# Test edge cases - single item corpus
build_index(model, ['C999'], ['Single concept'], repos='test_single')
results = search_similar(['Q999'], ['Single query'], top_k=5, repos='test_single')
assert isinstance(results, pd.DataFrame), "Should return DataFrame even with single concept"

# Test with top_k larger than corpus size
results = search_similar(query_ids, query_names, top_k=10, repos='test_build')
assert isinstance(results, pd.DataFrame), "Should handle top_k larger than corpus size"

# Clean up test repositories
for repo in ['test_build', 'test_error1', 'test_error2', 'test_precomputed', 'test_single']:
    delete_repository(repo)
    assert not is_initialized(repo), f"Repository {repo} should be deleted"

