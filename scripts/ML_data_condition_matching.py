# This file create training data for the ML model
# 
# Depends: data/ML/base_data/conceptML.feather
# 
# Sampling strategy: For each standard condition
# - Create positive samples by matching it to non-standard conditions
# - Create negative samples by randomly Sample n_neg negative samples
# - Create false positive samples in the file "scripts/ML_FP_condition.py"
# 
# Output: pd.DataFrame, 
# columns=['sentence1', 'sentence2', 'concept_id1', 'concept_id2']
# sentence1: standard concept name
# sentence2: non-standard concept name, synonym, or description
# concept_id1: concept_id of sentence1
# concept_id2: concept_id of sentence2 (if available)
#
# 
# Measure time consumption for each dataset
# Target: less than 10 minutes
#
# Note:
# - Standard concepts will include all concepts, regardless of whether they are reserved or not.
# - Non-standard concepts will only include non-standard concepts that are not reserved.
import os
import pandas as pd
import numpy as np
from modules.ML_sampling import generate_matching_positive_samples,\
    MatchingIterableDataset, get_sentence_name,\
    combine_columns,remove_reserved



concept = pd.read_feather('data/omop_feather/concept.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')
concept_ancestor = pd.read_feather('data/omop_feather/concept_ancestor.feather')
conceptML = pd.read_feather('data/ML/base_data/conceptML.feather')


conceptML.columns
# ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
#        'concept_code', 'nonstd_name', 'nonstd_concept_id', 'synonym_name',
#        'description', 'all_nonstd_name', 'all_nonstd_concept_id']



## Define the target standard concepts
## Reserved concepts are also in the target concepts!
std_target = conceptML[conceptML['domain_id'] == 'Condition'].reset_index(drop=True)
std_target['std_name'] = get_sentence_name(std_target['domain_id'], std_target['concept_name'])

print(f"std concept #: {len(std_target)}")   # 160288
print(std_target.columns)
# ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
#        'concept_code', 'nonstd_name', 'nonstd_concept_id', 'synonym_name',
#        'description', 'all_nonstd_name', 'all_nonstd_concept_id', 'source',
#        'std_name']



# Exclude the reserved concepts from the non-standard concepts.
# - This will NOT exclude the reserved concepts from the standard concepts as standard concepts are the required component in OMOP.
reserved_vocab = "CIEL"
reserved_concepts = concept[(concept.standard_concept != 'S')&(concept.vocabulary_id == reserved_vocab)]
reserved_concept_ids = set(reserved_concepts.concept_id.to_list())
print(f"reserved concepts #: {len(reserved_concept_ids)}") # 50881



# Calculate the total number of non-standard concepts
def total_nonstd(df):
    nonstd_concept_ids_list = df['all_nonstd_concept_id'].to_list()
    # expand the list
    nonstd_concept_ids_list = [concept_id for sublist in nonstd_concept_ids_list for concept_id in sublist]
    return len(nonstd_concept_ids_list)

print(f"nonstd # before removal: {total_nonstd(std_target)}") # 679651


## remove reserved concepts from std mapping
std_target[['nonstd_concept_id', 'nonstd_name']] = std_target.apply(
    lambda x: remove_reserved(x, reserved_concept_ids, 'nonstd_concept_id', 'nonstd_name'), axis=1, result_type='expand'
)

std_target[['all_nonstd_concept_id', 'all_nonstd_name']] = std_target.apply(
    lambda x: remove_reserved(x, reserved_concept_ids, 'all_nonstd_concept_id', 'all_nonstd_name'), axis=1, result_type='expand'
)

## total number of non-standard concepts after removing reserved concepts
print(f"nonstd # after removal: {total_nonstd(std_target)}") # 648802

##########################################
## Direct sentence  matching
##########################################

# Create positive samples test
positive_df_matching = generate_matching_positive_samples(
    df = std_target
    )

non_standard_count = positive_df_matching['sentence2'].nunique()  
print(f"Number of non-standard samples: {non_standard_count}") 
# Number of non-standard samples: 459321

positive_df_matching.iloc[0]
"""
sentence1      Condition: Infection present (Deprecated)
sentence2                              Infection present
concept_id1                                     40664135
concept_id2                                         <NA>
label                                                  1
source                                        all_nonstd
Name: 0, dtype: object
"""

# distribution of source 
print(positive_df_matching['source'].value_counts())
"""
do we need to make the positive samples balanced?
nonstd_name     440913
synonym_name    174363
descriptions     60286
"""


## Candidate: concept name, std_name, all_nonstd_name
## They share the same concept_id, which is the id for the standard concept!
candidate_df_matching = std_target[['concept_id', 'concept_name', 'std_name', 'all_nonstd_name']].copy()
candidate_df_matching['concept_name'] = combine_columns(
    candidate_df_matching[["concept_name", "std_name", "all_nonstd_name"]]
)
candidate_df_matching = candidate_df_matching[['concept_id', 'concept_name']].explode(
    'concept_name'
    )

print(f'Candidate df: {len(candidate_df_matching)}')  # 996138


# Test the iterable dataset
iterable_matching = MatchingIterableDataset(
    positive_df_matching = positive_df_matching,
    candidate_df_matching = candidate_df_matching,
    n_neg=4,  
    seed=42
)

it = iter(iterable_matching)
next(it)
"""
{'sentence1': 'Condition: Neoplasm, benign of hematopoietic system, NOS', 'sentence2': '[MATCHING] Lymphoid Benign (LBGN)', 'concept_id1': 751726, 'concept_id2': 777429, 'label': 1}
"""

## save objects
os.makedirs('data/ML/matching', exist_ok=True)
positive_df_matching.to_feather('data/ML/matching/positive_df_matching.feather')
candidate_df_matching.to_feather('data/ML/matching/candidate_df_matching.feather')
