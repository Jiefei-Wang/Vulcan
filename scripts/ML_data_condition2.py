# This file create training data for the ML model
# 
# Depends: data/ML/conceptML.feather
# 
# Sampling strategy:
# - For each non-standard condition, create positive samples
# - For each non-standard condition, sample n_neg(=4) negative samples
#
# Output: pd.DataFrame, 
# columns=['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label1', 'source']
# sentence1: standard concept name
# sentence2: non-standard concept name, synonym, or description
# concept_id1: concept_id of sentence1
# concept_id2: concept_id of sentence2 (if available)
# label1: 1 if sentence1 maps to sentence2, 0 otherwise
# source: The source of sentence2

# label1 = 1 if sentence1 maps to sentence2, 0 otherwise
# label2 = 1 if sentence1 is a parent of sentence2, 0 otherwise
# label3 = 1 if sentence1 is a child of sentence2, 0 otherwise


# Bert -> Embeddings -> FC(n input, n output) -> Cosine Similarity <- whether two sentences are similar
# positive: high similarity
# negative: low similarity
# loss1 = negative - positive
# 
# Bert -> Embeddings -> FC(n input, n output) -> Cosine Similarity <- whether two sentences have parent-child relationship
# positive: high similarity
# negative: low similarity
# loss2 = negative - positive
#
# total loss = loss1 + loss2 + loss3

from tqdm import tqdm
import pandas as pd
import random
from modules.ML_sampling import generate_positive_samples, generate_negative_samples, create_relation_maps

# Precise mapping, sentence1 maps to sentence 2

conceptML = pd.read_feather('data/ML/conceptML1.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')
concept_ancestor = pd.read_feather('data/omop_feather/concept_ancestor.feather')

## get standard conditions
std_condition = conceptML[conceptML['domain_id'] == 'Condition']
print(len(std_condition))   # 160288
print(std_condition.columns)

# filter the data which the following columns are not empty
# 'nonstd_name', 'synonym_name', 'descriptions'
std_target_with_nonstd = std_condition[
    std_condition['nonstd_name'].notna() | 
    std_condition['synonym_name'].notna() | 
    std_condition['descriptions'].notna()
].copy()
## Turn standard concept into a sentence for mapping
std_target_with_nonstd['std_name'] = std_target_with_nonstd['domain_id'] + ': ' + std_target_with_nonstd['concept_name']

## combine list of nonstd_name, synonym_name, and descriptions into a single column
std_target_with_nonstd["all_nonstd"] = std_target_with_nonstd[["nonstd_name", "synonym_name", "descriptions"]].apply(
    lambda x: [j for i in x if i is not None for j in i], axis=1
)

print(len(std_target_with_nonstd))  # 101896
print(std_target_with_nonstd.columns)
# Index(['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
#        'concept_code', 'nonstd_name', 'nonstd_concept_id', 'synonym_name',
#        'descriptions', 'std_name'],
#       dtype='object')


# Create positive samples test
positive_sample_dataset = generate_positive_samples(
    df = std_target_with_nonstd, 
    columns = ['nonstd_name', 'synonym_name', 'descriptions'],
    column_ids = ['nonstd_concept_id', None, None]
    )

non_standard_count = positive_sample_dataset['sentence2'].nunique()  
print(f"Number of non-standard samples: {non_standard_count}") #487682 unique non-standard samples
print(positive_sample_dataset.columns) # ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label1', 'source']

# distribution of source 
print(positive_sample_dataset['source'].value_counts(normalize=True))

"""
do we need to make the positive samples balanced?
nonstd_name     0.667803
synonym_name    0.246849
descriptions    0.085348
"""



target_ids = std_target_with_nonstd['concept_id']
relation_maps = create_relation_maps(concept_ancestor, target_ids)

len(relation_maps) # 101875
relation_maps.columns 
# ['from_concept_id', 'to_concept_id', 'min_levels_of_separation',
#        'max_levels_of_separation', 'type']



# negative sampling strategy:
# For each row in positive_sample_dataset
# - sample n_neg(=4) negative samples
# - negative samples are randomly selected from the standard concepts that are
# not in the parent/child list of the positive sample

n_neg = 4
negative_sample_dataset = generate_negative_samples(
    positive_samples = positive_sample_dataset,
    relation_maps = relation_maps,
    std_target_with_nonstd = std_target_with_nonstd,
    n_neg = n_neg,
    seed = 42
)


positive_sample_dataset.to_feather('data/ML/conceptML_positive.feather')
negative_sample_dataset.to_feather('data/ML/conceptML_negative.feather')
