import os
import pandas as pd
from modules.ModelFunctions import load_ST_model
from modules.CodeBlockExecutor import trace, tracedf
from modules.FalsePositives import get_false_positives


n_fp_matching = 50
base_path = "data/matching"
std_condition_concept = pd.read_feather(os.path.join(base_path, 'std_condition_concept.feather'))

target_concepts = std_condition_concept
tracedf(target_concepts)
#> DataFrame dimensions: 160288 rows × 10 columns
#> Column names:
#> ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id', 'concept_class_id', 'standard_concept', 'concept_code', 'valid_start_date', 'valid_end_date', 'invalid_reason']
#> Estimated memory usage: 68.51 MB



base_model = 'ClinicalBERT'
model, tokenizer = load_ST_model(base_model)

fp = get_false_positives(
    model=model,
    corpus_concepts=target_concepts,
    n_fp=n_fp_matching,
    repos='target_concepts_initial_model'
)
fp.to_feather(os.path.join(base_path, f'fp_matching_{n_fp_matching}.feather'))


tracedf(fp)
#> DataFrame dimensions: 7894103 rows × 6 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'score', 'label']
#> Estimated memory usage: 1.62 GB

