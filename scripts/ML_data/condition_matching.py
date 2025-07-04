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
    MatchingIterableDataset
from modules.timed_logger import logger
from modules.ML_data_condition_target_vars import *

logger.reset_timer()

##########################################
## Direct sentence  matching
##########################################
logger.log("Create positive samples")

all_targets = pd.read_feather('data/ML/all_targets.feather')
matching_map = pd.read_feather('data/ML/matching_map.feather')




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
source                                      synonym_name
Name: 0, dtype: object
"""

# distribution of source 
positive_df_matching['source'].value_counts()
"""
do we need to make the positive samples balanced?
nonstd_name     440913
synonym_name    174363
descriptions     60286
"""


logger.log("Create candidate df for negative samples")
## Candidate: concept name, std_name, all_nonstd_name
## They share the same concept_id, which is the id for the standard concept!
candidate_df_matching = std_target[['concept_id', 'concept_name', 'std_name', 'all_nonstd_name']].copy()
candidate_df_matching['concept_name'] = candidate_df_matching.apply(
    lambda x: [x['concept_name']] + [x['std_name']] + x['all_nonstd_name'], axis=1
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
{'sentence1': 'Condition: Infection present (Deprecated)', 'sentence2': 'Infection present', 'concept_id1': 40664135, 'concept_id2': <NA>, 'label': 1}
"""

logger.log("Save the positive and candidate df")
## save objects
os.makedirs('data/ML/matching', exist_ok=True)
positive_df_matching.to_feather('data/ML/matching/positive_df_matching.feather')
candidate_df_matching.to_feather('data/ML/matching/candidate_df_matching.feather')

logger.done()