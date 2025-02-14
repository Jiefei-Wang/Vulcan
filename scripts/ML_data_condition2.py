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
# 1. Matching dataset
# 2. Relation dataset
# 
# Measure time consumption for each dataset
# Target: less than 10 minutes
#
# Iterable Class: params: Number of negative samples
# 

import pandas as pd
from modules.ML_sampling import generate_matching_positive_samples, generate_matching_negative_samples, get_filtered_concept_ancestor, generate_relation_positive_samples, generate_relation_negative_samples,get_sentence_name, add_special_token_df

# Precise mapping, sentence1 maps to sentence 2

concept = pd.read_feather('data/omop_feather/concept.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')
concept_ancestor = pd.read_feather('data/omop_feather/concept_ancestor.feather')


conceptML = pd.read_feather('data/ML/conceptML1.feather')

## Define the target standard concepts
std_target = conceptML[conceptML['domain_id'] == 'Condition'].reset_index(drop=True)
std_target['std_name'] = get_sentence_name(std_target['domain_id'], std_target['concept_name'])
print(len(std_target))   # 160288
print(std_target.columns)


target_standard_ids = std_target['concept_id']
## Filter concept_ancestor to only include the target standard concepts
## in either ancestor_concept_id or descendant_concept_id
concept_ancestor_filtered = get_filtered_concept_ancestor(concept_ancestor, target_standard_ids)

##########################################
## Direct sentence  matching
##########################################
## combine list of nonstd_name, synonym_name, and descriptions into a single column
std_target_for_matching = std_target.copy()
std_target_for_matching["all_nonstd"] = std_target_for_matching[["nonstd_name", "synonym_name", "descriptions"]].apply(
    lambda x: [j for i in x if i is not None for j in i], axis=1
)
## filter out rows with no non-standard names
std_target_for_matching = std_target_for_matching[std_target_for_matching['all_nonstd'].str.len() > 0]

print(len(std_target_for_matching))  # 101896
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
print(f"Number of non-standard samples: {non_standard_count}") #487682 unique non-standard samples

positive_dataset_matching.iloc[0]
"""
sentence1      Condition: Neoplasm, benign of hematopoietic s...
sentence2                                 Lymphoid Benign (LBGN)
concept_id1                                               751726
concept_id2                                               777429
label                                                          1
source                                               nonstd_name
"""

# distribution of source 
print(positive_dataset_matching['source'].value_counts(normalize=True))
"""
do we need to make the positive samples balanced?
nonstd_name     0.667803
synonym_name    0.246849
descriptions    0.085348
"""


# negative sampling strategy:
# For each row in positive_sample_dataset
# - sample n_neg(=4) negative samples
# - negative samples are randomly selected from the standard concepts that are
# not in the parent/child list of the positive sample

n_neg = 4
negative_dataset_matching = generate_matching_negative_samples(
    positive_dataset_matching = positive_dataset_matching,
    concept_ancestor = concept_ancestor,
    std_target_for_matching = std_target_for_matching,
    n_neg = n_neg,
    seed = 42
)

len(negative_dataset_matching) # 2825424
print(negative_dataset_matching.columns) # ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label', 'source']

negative_dataset_matching.iloc[0]
"""
sentence1      Condition: Neoplasm, benign of hematopoietic s...
sentence2                          Atheroma of Cerebral Arteries
concept_id1                                               751726
concept_id2                                              4009154
label                                                          0
source                                         matching_negative
"""

##########################################
## Label2: ancestor-descendant relationship
## sentence 1: ancestor concept
## sentence 2: descendant concept
##########################################
positive_dataset_relation = generate_relation_positive_samples(concept_ancestor_filtered, concept)

positive_dataset_relation = positive_dataset_relation[['sentence1',
       'sentence2', 'concept_id1', 'concept_id2', 'label']]

positive_dataset_relation.columns
len(positive_dataset_relation) # 2819798

print(positive_dataset_relation.iloc[0])
"""
sentence1                         Condition: Lesion of genitalia
sentence2      Condition: T-cell large granular lymphocytic l...
concept_id1                                             37110208
concept_id2                                             36561313
label                                                          1
"""

## negative samples
negative_dataset_relation = generate_relation_negative_samples(positive_dataset_relation, concept_ancestor_filtered, concept, n_neg=4)

len(negative_dataset_relation) # 11279192


negative_dataset_relation.iloc[0]
'''
sentence1              Condition: Lesion of genitalia
sentence2      Observation: Suprapatellar jerk absent
concept_id1                                  37110208
concept_id2                                   3661505
label                                               0
Name: 0, dtype: object
'''

positive_dataset_matching.to_feather('data/ML/conceptML_positive_matching.feather')
negative_dataset_matching.to_feather('data/ML/conceptML_negative_matching.feather')
positive_dataset_relation.to_feather('data/ML/conceptML_positive_relation.feather')
negative_dataset_relation.to_feather('data/ML/conceptML_negative_relation.feather')




##################
## Prepare training and validation dataset
##################
positive_dataset_matching = pd.read_feather('data/ML/conceptML_positive_matching.feather')
negative_dataset_matching = pd.read_feather('data/ML/conceptML_negative_matching.feather')
positive_dataset_relation = pd.read_feather('data/ML/conceptML_positive_relation.feather')
negative_dataset_relation = pd.read_feather('data/ML/conceptML_negative_relation.feather')

positive_matching = add_special_token_df(positive_dataset_matching, '[MATCHING]')
negative_matching = add_special_token_df(negative_dataset_matching, '[MATCHING]')
positive_relation = add_special_token_df(positive_dataset_relation, '[RELATION]')
negative_relation = add_special_token_df(negative_dataset_relation, '[RELATION]')


matching_all = pd.concat(
    [positive_matching, negative_matching]
    ).reset_index(drop=True)

relation_all = pd.concat(
    [positive_relation, negative_relation]
    ).reset_index(drop=True)

column_keep = ['sentence1', 'sentence2', 'label']
## Check for missing values
matching_all[matching_all[column_keep].isnull().any(axis=1)]
relation_all[relation_all[column_keep].isnull().any(axis=1)]


print(len(matching_all)) # 3531780
print(len(relation_all)) # 14684495

##################
## Validation dataset
##################
reserved_vocab = "CIEL"
reserved_concepts = concept[(concept.domain_id=="Condition")&(concept.vocabulary_id == reserved_vocab)]
print(len(reserved_concepts)) # 38818
reserved_concept_ids = set(reserved_concepts.concept_id.to_list())

## filter out reserved vocab from column concept_id1 and concept_id2
def filter_out_reserved(df, reserved_ids):
    return df[(~df.concept_id1.isin(reserved_ids)) & (~df.concept_id2.isin(reserved_ids))]

matching_filtered = filter_out_reserved(matching_all, reserved_concept_ids)
relation_filtered = filter_out_reserved(relation_all, reserved_concepts.concept_id)

## reorder rows
matching_filtered = matching_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
relation_filtered = relation_filtered.sample(frac=1, random_state=42).reset_index(drop=True)

print(len(matching_filtered)) # 3500986
print(len(relation_filtered)) # 14684495

matching_filtered[column_keep].to_feather('data/ML/conceptML_matching.feather')
relation_filtered[column_keep].to_feather('data/ML/conceptML_relation.feather')

## make validation dataset
def filter_in_reserved(df, reserved_ids):
    return df[(df.concept_id1.isin(reserved_ids)) | (df.concept_id2.isin(reserved_ids))]

matching_validation = filter_in_reserved(matching_all, reserved_concept_ids)

print(len(matching_validation)) # 30794

matching_validation[column_keep].to_feather('data/ML/conceptML_matching_validation.feather')