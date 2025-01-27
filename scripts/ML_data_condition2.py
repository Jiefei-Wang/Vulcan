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
from modules.ML_sampling import generate_positive_samples, generate_negative_samples, create_relation_maps, generate_parent_child_positive_samples, generate_negative_parent_child_samples

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

"""
create label 2 dataset 
"""

# # Extract all unique concept IDs from relation map to map with their names in the standard concept table
# 
# use conceptML to extract all the standard concepts 
# Step 1: Extract all unique concept IDs from relation_maps
relation_maps_concept_ids = relation_maps.explode("to_concept_id")
concept_ids_to_extract = set(relation_maps_concept_ids["from_concept_id"]).union(set(relation_maps_concept_ids["to_concept_id"]))

# Convert concept IDs to DataFrame
all_concept_ids = pd.DataFrame({"concept_id": list(concept_ids_to_extract)})

# Step 2: Filter concept names from std_target_with_nonstd
filtered_concept_id_name_df = all_concept_ids.merge(
    conceptML[['concept_id', 'concept_name']], 
    on="concept_id", 
    how="inner"
)

# missing_ids = set(all_concept_ids["concept_id"]) - set(filtered_concept_id_name_df["concept_id"])
# print("Missing IDs:", missing_ids)
# print(f"Total missing IDs: {len(missing_ids)}")

print(len(concept_ids_to_extract)) # 168816
# all standard concepts with their names which is in both ConceptML and relation_maps
print(len(filtered_concept_id_name_df)) # 168773 [concept_id, concept_name]
# # 43 id doesn't belong to standard concept (not in the conceptML)


print(filtered_concept_id_name_df.iloc[0])
print(relation_maps.iloc[0])
# create exploded relation maps
relation_maps_expanded_pairs = relation_maps.explode(["to_concept_id", "min_levels_of_separation", "max_levels_of_separation", "type"])
print(relation_maps_expanded_pairs.columns)
print(len(relation_maps_expanded_pairs)) # 4095293
print(relation_maps_expanded_pairs.columns)


# Generate parent-child samples
positive_parent_child_samples = generate_parent_child_positive_samples(relation_maps_expanded_pairs, filtered_concept_id_name_df, relationship_type="ancestor")
print(len(positive_parent_child_samples)) # 2701039
print(positive_parent_child_samples.columns) 
print(positive_parent_child_samples.iloc[0])
print(positive_parent_child_samples[['parent_name', 'child_name', 'parent_id', 'child_id']].isna().sum())

"""
Index(['parent_name', 'child_name', 'parent_id', 'child_id',
       'min_levels_of_separation', 'max_levels_of_separation'],
      dtype='object')
"""
print(positive_parent_child_samples.iloc[0])
"""
parent_name                          Neoplasm of uncertain behavior of larynx
child_name                  Post-transplant lymphoproliferative disorder, ...
parent_id                                                               22274
child_id                                                              1553606
min_levels_of_separation                                                    1
max_levels_of_separation                                                    1
direction                                                     parent_to_child
"""

# create negative samples for parent-child relationship
# negative sampling strategy:
# For each row in relation_maps
# - sample n_neg(=4) negative samples
# - negative samples are randomly selected from the standard concepts 

relation_maps_expanded_pairs2 = relation_maps_expanded_pairs.rename(
    columns={"from_concept_id": "parent_id", "to_concept_id": "child_id"}
)
print(relation_maps_expanded_pairs2.columns)
"""
Index(['parent_id', 'child_id', 'min_levels_of_separation',
       'max_levels_of_separation', 'type'],
      dtype='object')
"""


print(len(relation_maps_expanded_pairs2)) # 4095293
print(len(positive_parent_child_samples)) # 2701039

negative_parent_child_samples = generate_negative_parent_child_samples(
    positive_parent_child=positive_parent_child_samples, 
    relation_maps_expanded_pairs=relation_maps_expanded_pairs2, 
    all_concept_ids=filtered_concept_id_name_df,  # Concept ID & Name Mapping
    relationship_type = "ancestor",
    n_neg=4
)
print(negative_parent_child_samples[['parent_name', 'child_name', 'parent_id', 'child_id']].isna().sum())


print(len(positive_sample_dataset)) 
print(positive_sample_dataset.columns)
print(len(negative_parent_child_samples))
print(negative_parent_child_samples.iloc[0])
"""
>>> print(negative_parent_child_samples.iloc[0])
parent_name             Neoplasm of uncertain behavior of larynx
child_name     Fibroblastic liposarcoma of long bones of uppe...
parent_id                                                  22274
child_id                                                36554817
label2                                                         0
source                                                  negative
Name: 0, dtype: object
"""

"""
create label 3 dataset 
"""
# same as the parent_child, with relationship_type="offspring"
positive_child_parent_samples = generate_parent_child_positive_samples(relation_maps_expanded_pairs, filtered_concept_id_name_df, relationship_type="offspring")


print(positive_child_parent_samples.columns)
print(positive_child_parent_samples.iloc[0])
print(len(positive_child_parent_samples)) # 1388795
print(positive_child_parent_samples[['parent_name', 'child_name', 'parent_id', 'child_id']].isna().sum())


# print(len(positive_child_parent_samples[positive_child_parent_samples['parent_id'] == 4111017])) # 1676
# print(relation_maps_expanded_pairs[relation_maps_expanded_pairs['to_concept_id'] == 4111017]) # 1680 (the difference is hierarchy level = 0)

# create negative samples for child-parent relationship
negative_child_parent_samples = generate_negative_parent_child_samples(
    positive_parent_child=positive_child_parent_samples, 
    relation_maps_expanded_pairs=relation_maps_expanded_pairs2, 
    all_concept_ids=filtered_concept_id_name_df,  # Concept ID & Name Mapping
    relationship_type = "offspring",
    n_neg=4
)

print(negative_child_parent_samples[["parent_name", "child_name"]].isin(["Unknown"]).sum()) # [0, 0]

print(len(negative_child_parent_samples)) # 5555180
print(negative_child_parent_samples.columns)
print(negative_child_parent_samples.iloc[1])
print(negative_child_parent_samples[['parent_name', 'child_name', 'parent_id', 'child_id']].isna().sum())



positive_sample_dataset.to_feather('data/ML/conceptML_positive.feather')
negative_sample_dataset.to_feather('data/ML/conceptML_negative.feather')
