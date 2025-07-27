import pandas as pd
import numpy as np
from modules.ModelFunctions import get_ST_model
from modules.FalsePositives import getFalsePositives
from modules.FaissDB import delete_repository

# Load model for testing
base_model = 'ClinicalBERT'
model, tokenizer = get_ST_model(base_model)

# Test data
corpus_ids = ['C001', 'C002', 'C003', 'C004', 'C005']
corpus_names = [
    'Diabetes mellitus',
    'Hypertension', 
    'Heart failure',
    'Chronic kidney disease',
    'Pneumonia'
]

# Test basic functionality with same corpus and query
fp_results = getFalsePositives(
    model=model,
    corpus_ids=corpus_ids,
    corpus_names=corpus_names,
    n_fp=3,
    repos='test_fp'
)

assert isinstance(fp_results, pd.DataFrame), "getFalsePositives should return a DataFrame"

expected_columns = ['query_id', 'query_name', 'corpus_id', 'corpus_name', 'score', 'label']
assert all(col in fp_results.columns for col in expected_columns), f"Missing columns. Expected: {expected_columns}, Got: {list(fp_results.columns)}"

assert all(fp_results['label'] == 0), "All results should have label=0 for false positives"
assert len(fp_results) <= len(corpus_ids) * 3, "Should return at most n_fp results per query"

# Test with different query set
query_ids = ['Q001', 'Q002']
query_names = ['Diabetes', 'High blood pressure']

fp_results2 = getFalsePositives(
    model=model,
    corpus_ids=corpus_ids,
    corpus_names=corpus_names,
    query_ids=query_ids,
    query_names=query_names,
    n_fp=2,
    repos='test_fp2'
)

assert isinstance(fp_results2, pd.DataFrame), "Should return DataFrame with different query set"
assert len(fp_results2) <= len(query_ids) * 2, "Should return at most n_fp results per query"
assert all(fp_results2['label'] == 0), "All results should have label=0"

# Test with blacklist
blacklist_from = ['C001']
blacklist_to = ['Q001']

fp_results3 = getFalsePositives(
    model=model,
    corpus_ids=corpus_ids,
    corpus_names=corpus_names,
    query_ids=query_ids,
    query_names=query_names,
    blacklist_from=blacklist_from,
    blacklist_to=blacklist_to,
    n_fp=2,
    repos='test_fp3'
)

# Should not contain the blacklisted pair
blacklisted_pairs = fp_results3[
    (fp_results3['corpus_id'] == 'C001') & 
    (fp_results3['query_id'] == 'Q001')
]
assert len(blacklisted_pairs) == 0, "Blacklisted pairs should be excluded"

# Test error cases
# Test mismatched corpus lengths
error_caught = False
try:
    getFalsePositives(
        model=model,
        corpus_ids=['C001'],
        corpus_names=['Name1', 'Name2'],
        repos='test_error1'
    )
except ValueError:
    error_caught = True
assert error_caught, "Should raise ValueError for mismatched corpus lengths"

# Test mismatched query lengths  
error_caught = False
try:
    getFalsePositives(
        model=model,
        corpus_ids=corpus_ids,
        corpus_names=corpus_names,
        query_ids=['Q001'],
        query_names=['Name1', 'Name2'],
        repos='test_error2'
    )
except ValueError:
    error_caught = True
assert error_caught, "Should raise ValueError for mismatched query lengths"

# Test mismatched blacklist lengths
error_caught = False
try:
    getFalsePositives(
        model=model,
        corpus_ids=corpus_ids,
        corpus_names=corpus_names,
        blacklist_from=['C001'],
        blacklist_to=['Q001', 'Q002'],
        repos='test_error3'
    )
except ValueError:
    error_caught = True
assert error_caught, "Should raise ValueError for mismatched blacklist lengths"

# Test with empty blacklist (should work fine)
fp_results4 = getFalsePositives(
    model=model,
    corpus_ids=corpus_ids,
    corpus_names=corpus_names,
    blacklist_from=[],
    blacklist_to=[],
    n_fp=1,
    repos='test_empty_blacklist'
)
assert isinstance(fp_results4, pd.DataFrame), "Should handle empty blacklist"

# Test with single concept
fp_results5 = getFalsePositives(
    model=model,
    corpus_ids=['C999'],
    corpus_names=['Single concept'],
    n_fp=5,
    repos='test_single'
)
assert isinstance(fp_results5, pd.DataFrame), "Should handle single concept"

# Clean up test repositories
repos_to_clean = ['test_fp', 'test_fp2', 'test_fp3', 'test_error1', 'test_error2', 'test_error3', 'test_empty_blacklist', 'test_single']
for repo in repos_to_clean:
    delete_repository(repo)

