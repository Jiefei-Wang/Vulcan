# This file create training data for the ML model
# 
# Depends: data/ML/conceptML.feather
# 
# Sampling strategy:
# - For each non-standard condition, create positive samples
# - For each non-standard condition, sample n_neg(=4) negative samples
#
# Output: pd.DataFrame, 
# columns=['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']
# sentence1: standard concept name
# sentence2: non-standard concept name, synonym, or description
# concept_id1: concept_id of sentence1
# concept_id2: concept_id of sentence2 (if available)
#
# For sentence matching:
#   - label = 1 if sentence1 maps to sentence2, 0 otherwise
# For parent-child relationship:
#   - label = 1 if sentence1 has relationship with sentence2, 0 otherwise


# Two Iterable Dataset
# 1. Matching dataset precise matching (label 0)
# 2. Relation dataset parent-child relationship (label 1)
# 
# Measure time consumption for each dataset
# Target: less than 10 minutes
#
# Iterable Class: params: Number of negative samples
# 
import os
import pandas as pd
import numpy as np
from modules.ML_sampling import generate_matching_positive_samples, get_filtered_concept_ancestor,\
    generate_relation_positive_samples, get_sentence_name,\
    add_special_token_df, OffspringIterableDataset, AncestorIterableDataset, MatchingIterableDataset, add_special_token, get_blacklist_map,\
    combine_columns,remove_reserved,replace_std_concept_with_reserved
from datasets import Dataset
import pyarrow.feather as feather

# relaod kernel to get the latest version of the modules
# Code has been moved to scripts/reload_library.py


# Precise mapping, sentence1 maps to sentence 2

concept = pd.read_feather('data/omop_feather/concept.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')
concept_ancestor = pd.read_feather('data/omop_feather/concept_ancestor.feather')


conceptML = pd.read_feather('data/ML/conceptML1.feather')

reserved_vocab = "CIEL"


## Define the target standard concepts
std_target = conceptML[(conceptML['domain_id'] == 'Condition')& (conceptML['vocabulary_id'] != reserved_vocab)].reset_index(drop=True)
std_target['std_name'] = get_sentence_name(std_target['domain_id'], std_target['concept_name'])
print(len(std_target))   # 160288
print(std_target.columns)
# ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
#        'concept_code', 'nonstd_name', 'nonstd_concept_id', 'synonym_name',
#        'descriptions', 'std_name']

## For some reserved vocab, they are not standard concepts
reserved_concepts = concept[(concept.domain_id=="Condition")&(concept.vocabulary_id == reserved_vocab)]
reserved_concept_ids = set(reserved_concepts.concept_id.to_list())


## remove reserved concepts from 'nonstd_name', 'nonstd_concept_id' list
std_target[['nonstd_concept_id', 'nonstd_name']] = std_target.apply(
    lambda x: remove_reserved(x, reserved_concept_ids), axis=1, result_type='expand'
)

##########################################
## Direct sentence  matching
##########################################
## combine list of nonstd_name, synonym_name, and descriptions into a single column
std_target_for_matching = std_target.copy()
std_target_for_matching["all_nonstd"] = combine_columns(
    std_target_for_matching[["nonstd_name", "synonym_name", "descriptions"]])



## filter out rows with no non-standard names
print(len(std_target_for_matching))  # 160288
print(std_target_for_matching.columns)
"""
Index(['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
       'concept_code', 'nonstd_name', 'nonstd_concept_id', 'synonym_name',
       'descriptions', 'std_name', 'all_nonstd'],
      dtype='object')
"""

# Create positive samples test
positive_dataset_matching = generate_matching_positive_samples(
    df = std_target_for_matching, 
    columns = ['nonstd_name', 'synonym_name', 'descriptions'],
    column_ids = ['nonstd_concept_id', None, None]
    )


non_standard_count = positive_dataset_matching['sentence2'].nunique()  
print(f"Number of non-standard samples: {non_standard_count}") 
# Number of non-standard samples: 459427

positive_dataset_matching.iloc[0]
"""
sentence1      Condition: Neoplasm, benign of hema...
sentence2                      Lymphoid Benign (LBGN)
concept_id1                                               751726
concept_id2                                             777429.0
label                                                          1
source                                               nonstd_name
Name: 0, dtype: object
"""

# distribution of source 
print(positive_dataset_matching['source'].value_counts())
"""
do we need to make the positive samples balanced?
nonstd_name     440913
synonym_name    174363
descriptions     60286
"""


candidate_df_matching = std_target_for_matching[['concept_id', 'concept_name', 'std_name', 'all_nonstd']].copy()
candidate_df_matching['concept_name'] = combine_columns(
    std_target_for_matching[["concept_name", "std_name", "all_nonstd"]]
)
candidate_df_matching = candidate_df_matching[['concept_id', 'concept_name']].explode(
    'concept_name'
    )
len(candidate_df_matching)  # 996138

# Test the iterable dataset
iterable_matching = MatchingIterableDataset(
    positive_dataset_matching,
    candidate_df_matching,
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
positive_dataset_matching.to_feather('data/ML/matching/positive_dataset_matching.feather')
candidate_df_matching.to_feather('data/ML/matching/candidate_dataset_matching.feather')


##########################################
## Label2: ancestor-descendant relationship
## sentence 1: ancestor concept
## sentence 2: descendant concept
##########################################

## Keep those concept relationship that exists in target_standard_ids
target_standard_ids = std_target['concept_id']
concept_ancestor_filtered = get_filtered_concept_ancestor(concept_ancestor, target_standard_ids)
positive_dataset_relation = generate_relation_positive_samples(concept_ancestor_filtered, concept)
positive_dataset_relation = positive_dataset_relation[['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']]
len(positive_dataset_relation) # 2936899


n_neg=4
seed=123
myiter = OffspringIterableDataset(
    positive_dataset_relation,
    std_target
)

it = iter(myiter)
next(it)
"""
{'sentence1': '[OFFSPRINT] Condition: Lesion of genitalia', 'sentence2': 'Condition: T-cell large granular lymphocytic leukemia of labium majus', 'concept_id1': 37110208, 'concept_id2': 36561313, 'label': 1}
"""

## Save data
os.makedirs('data/ML/relation', exist_ok=True)
positive_dataset_relation.to_feather('data/ML/relation/positive_dataset_relation.feather')
std_target.to_feather('data/ML/relation/std_target.feather')

##########################
## Find ancestor concept by adding [ANCESTOR] token
##########################
myiter = AncestorIterableDataset(
    positive_dataset_relation,
    std_target
)

it = iter(myiter)
next(it)

"""
{'sentence1': 'Condition: Lesion of genitalia', 'sentence2': '[ANCESTOR] Condition: T-cell large granular lymphocytic 
leukemia of labium majus', 'concept_id1': 37110208, 'concept_id2': 36561313, 'label': 1}
"""


##################
## Validation dataset
##################
reserved_vocab = "CIEL"
reserved_concepts = concept[(concept.domain_id=="Condition")&(concept.vocabulary_id == reserved_vocab)]
print(len(reserved_concepts)) # 38818

reserved_concept_ids = set(reserved_concepts.concept_id.to_list())

## Make sure we can map the reserved concept back to standard concepts
to_std = concept_relationship[concept_relationship['relationship_id'] == 'Maps to']
std_bridge = to_std[to_std.concept_id_1.isin(reserved_concept_ids)].rename({'concept_id_1': 'concept_id1', 'concept_id_2': 'concept_id2'}, axis=1)[['concept_id1', 'concept_id2']]

reserved_concepts = reserved_concepts[reserved_concepts['concept_id'].isin(std_bridge.concept_id1.to_list())]
print(len(reserved_concepts)) # 34217

## construct validation dataset
positive_matching_validation = std_bridge.merge(
    concept[['concept_id', 'concept_name']],
    left_on='concept_id1',
    right_on='concept_id',
    how='inner'
).rename({'concept_name': 'sentence1'}, axis=1
).drop('concept_id', axis=1
).merge(
    concept[['concept_id', 'concept_name']],
    left_on='concept_id2',
    right_on='concept_id',
    how='inner'
).rename({'concept_name': 'sentence2'}, axis=1
).drop('concept_id', axis=1)

positive_matching_validation['label'] = 1


candidate_matching_validation = std_target[['concept_id', 'std_name']].rename({'std_name': 'concept_name'}, axis=1)


os.makedirs('data/ML/validation', exist_ok=True)
positive_matching_validation.to_feather('data/ML/validation/positive_matching_validation.feather')
candidate_matching_validation.to_feather('data/ML/validation/candidate_matching_validation.feather')

iterable_matching_validation = MatchingIterableDataset(
    positive_matching_validation,
    candidate_matching_validation,
    n_neg=1,  
    seed=42,
    special_token_sentence1=True,
    special_token_sentence2=False,
    special_token_candidate=False
)

it = iter(iterable_matching_validation)
next(it)
"""
'sentence1': '[MATCHING] Incomplete Illegally Induced Abortion Complicated by Genital Tract and Pelvic Infection', 
'sentence2': 'Induced termination of pregnancy complicated by genital-pelvic infection', 'concept_id1': 45913718, 'concept_id2': 43530910, 'label': 1
"""


# For offspring
reserved_std_ids = std_bridge['concept_id2'].to_list()
concept_ancestor_validation = get_filtered_concept_ancestor(concept_ancestor, reserved_std_ids)
concept_ancestor_validation = replace_std_concept_with_reserved(concept_ancestor_validation, std_bridge, 'ancestor_concept_id')
concept_ancestor_validation = replace_std_concept_with_reserved(concept_ancestor_validation, std_bridge, 'descendant_concept_id')
concept_ancestor_validation = concept_ancestor_validation.drop_duplicates()
len(concept_ancestor_validation) # 3253525



positive_dataset_relation_validation = generate_relation_positive_samples(concept_ancestor_validation, concept)

positive_dataset_relation_validation.iloc[0]

len(positive_dataset_relation_validation) # 2936899


## for matching to offspring and ancestor
positive_dataset_to_offspring_validation = positive_dataset_relation_validation[
    positive_dataset_relation_validation['concept_id1'].isin(reserved_concept_ids)
][['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']]

positive_dataset_to_offspring_validation['sentence1'] = positive_dataset_to_offspring_validation['sentence1'].apply(
    lambda x: f"[offspring] {x}"
)


positive_dataset_to_ancestor_validation = positive_dataset_relation_validation[
    positive_dataset_relation_validation['concept_id2'].isin(reserved_concept_ids)
][['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']]

positive_dataset_to_ancestor_validation['sentence2'] = positive_dataset_to_ancestor_validation['sentence2'].apply(
    lambda x: f"[ancestor] {x}"
)

positive_dataset_to_offspring_validation.to_feather('data/ML/validation/positive_dataset_to_offspring_validation.feather')
positive_dataset_to_ancestor_validation.to_feather('data/ML/validation/positive_dataset_to_ancestor_validation.feather')