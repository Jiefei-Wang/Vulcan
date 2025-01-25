# This file create training data for the ML model
# 
# Depends: data/ML/conceptML.feather
# 
# Sampling strategy:
# - For each non-standard condition, create positive samples
# - For each non-standard condition, sample n_neg(=4) negative samples
#
# Output: pd.DataFrame, 
# columns=['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label1']
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
import numpy as np
import random
from itertools import islice
import time

# Precise mapping, sentence1 maps to sentence 2

conceptML = pd.read_feather('data/ML/conceptML1.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')

## get standard conditions
std_condition = conceptML[conceptML['domain_id'] == 'Condition']
print(len(std_condition))   # 160288
print(std_condition.columns)

#filter the data which nonstd_concept is not empty
std_condition_with_nonstd = std_condition[
    std_condition['nonstd_concept_id'].apply(lambda x: x is not None and len(x) > 0)
]
print(len(std_condition_with_nonstd))  # 83763

"""
# Training data structure:
# sentence1: non-standard concept name, synonym, or UMLS description
# sentence2: standard concept name
# - std_name: "Condition: concept_name"
# - concept_synonym_name: list of synonyms
# - umls_description: list of UMLS descriptions

# add new column to std_condition_with_nonstd
# new column: std_name: "Condition: concept_name"
"""

std_condition_with_nonstd_updated = std_condition_with_nonstd.copy()
std_condition_with_nonstd_updated.loc[:, 'std_name'] = std_condition_with_nonstd['domain_id'] + ': ' + std_condition_with_nonstd['concept_name']
print(std_condition_with_nonstd_updated.columns)
"""
Index(['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
       'concept_code', 'nonstd_name', 'nonstd_concept_id',
       'concept_synonym_name', 'umls_desc', 'std_name'],
      dtype='object')
"""

"""
# Positive Sample Generation
# For each non-standard condition, create positive samples that map to a standard concept.
# Steps:
# - Use the non-standard condition name (nonstd_name) and its associated nonstd_concept_id.
# - Map it to the corresponding standard condition name (std_concept).
# Structure:
# - sentence1: non-standard condition name
# - sentence2: standard condition name
# Function: Processes each row to generate positive samples.
# For every non-standard condition concept name, the positive samples of it includes the 
# non-standard condition name, synonym, and UMLS description.
"""

def generate_positive_samples(df, columns):
    """
    Process a dataset with multiple list-like or NoneType columns and add additional metadata.

    Args:
        df (pd.DataFrame): The input DataFrame with list-like or NoneType columns.
        columns (list of str): The columns to process (e.g., ['nonstd_name', 'concept_synonym_name', 'umls_desc']).

    Returns:
        pd.DataFrame: A processed dataset with exploded rows and additional metadata.
    """
    result_frames = []
    for column in columns:
        df[column] = df[column].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else ([] if x is None else x))
        exploded_df = df[['concept_name', 'concept_id', column, 'nonstd_concept_id']].explode(column)
        exploded_df['sentence1'] = f"Type: {column}, " + exploded_df[column].fillna("")
        exploded_df['sentence2'] = "Condition: " + exploded_df['concept_name']
        exploded_df['concept_id1'] = exploded_df['nonstd_concept_id'].apply(
                    lambda x: int(x[0]) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else (
                            int(x) if isinstance(x, (int, float)) else None
                            )
                    ) if column == 'nonstd_name' else None
        exploded_df['concept_id2'] = exploded_df['concept_id']
        exploded_df['label1'] = 1
        exploded_df = exploded_df[exploded_df[column].notna() & (exploded_df[column] != "")]
        result_frames.append(exploded_df[['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label1']])
    final_dataset = pd.concat(result_frames, ignore_index=True).drop_duplicates()
    return final_dataset

# Create positive samples test
positive_sample_dataset = generate_positive_samples(std_condition_with_nonstd_updated, ['nonstd_name', 'concept_synonym_name', 'umls_desc'])

non_standard_count = positive_sample_dataset['sentence1'].nunique()  
print(f"Number of non-standard samples: {non_standard_count}") #475033 unique non-standard samples (might too large)

print(type(positive_sample_dataset)) # <class 'pandas.core.frame.DataFrame'>
print(positive_sample_dataset.columns) # ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label1']

# protion of positive samples
positive_sample_dataset['type'] = positive_sample_dataset['sentence1'].apply(lambda x: 'nonstd_name' if 'nonstd_name' in x else ('umls_desc' if 'umls_desc' in x else 'synonym'))
print(positive_sample_dataset['type'].value_counts(normalize=True))


"""
do we need to make the positive samples balanced?
nonstd_name    0.623697
synonym        0.273176
umls_desc      0.103127
"""

# create dictionary for positive samples for quick lookup for positive samples to prevent overlap with the negative samples
positive_dict = {}

for _, row in positive_sample_dataset.iterrows():
    key = row['sentence1']  # Use sentence1 as the key
    value = {
        "sentence2": row['sentence2'],
        "concept_id1": row['concept_id1'],
        "concept_id2": row['concept_id2'],
        "label1": row['label1']
    }
    # Append to the list of values for this key
    if key not in positive_dict:
        positive_dict[key] = []
    positive_dict[key].append(value)


"""
The type of the non-standard condition:
Extracted Types: {'Type: umls_desc', 'Type: concept_synonym_name', 'Type: nonstd_name'}
"""

"""
# Negative sample generation 
# For each non-standard condition, sample n_neg(=4) negative samples

"""
# generate negative samples
def generate_negative_samples(positive_dict, std_concepts, n_neg=4):
    """
    Generate negative samples from standard concepts for each sentence1 in the positive dictionary.

    Args:
        positive_dict (dict): Dictionary of positive samples, where keys are `sentence1`
                             and values are lists of dictionaries containing `sentence2` and metadata.
        std_concepts (dict): Dictionary of standard concepts with `sentence2` as keys
                            and `concept_id2` as values.
        n_neg (int): Number of negative samples to generate per `sentence1`. Default is 4.

    Returns:
        list: List of negative samples, where each sample is a dictionary with
              `sentence1`, `sentence2`, `concept_id1`, `concept_id2`, and `label1`.
    """
    negative_samples = []
    all_sentence2 = set(std_concepts.keys())  # Use only standard concept names
    for sentence1, positive_values in tqdm(positive_dict.items(), desc="Generating Negative Samples"):
        positive_sentence2 = {value["sentence2"] for value in positive_values}
        available_negatives = list(all_sentence2 - positive_sentence2)
        selected_negatives = random.sample(available_negatives, min(n_neg, len(available_negatives)))
        for neg_sentence2 in selected_negatives:
            negative_samples.append({
                "sentence1": sentence1,
                "sentence2": neg_sentence2,
                "concept_id1": positive_values[0]["concept_id1"] if positive_values else None,
                "concept_id2": std_concepts[neg_sentence2],  # Get ID from the std_concepts dictionary
                "label1": 0  # Negative label
            })
    return negative_samples

## create a std-concept with its id dictionary for quick lookup
# key: Condition: std condition name
# values: concept_id
std_concepts = {}

for sentence1, values in positive_dict.items():
    for value in values:
        sentence2 = value["sentence2"]
        concept_id2 = value["concept_id2"]
        if concept_id2 is not None:  # Only include standard concepts with IDs
            std_concepts[sentence2] = concept_id2

negative_samples = generate_negative_samples(positive_dict, std_concepts, n_neg=4)
print(negative_samples) 






