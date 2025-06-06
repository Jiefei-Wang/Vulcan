# Create parent-child relation for model training
import os
import pandas as pd
from modules.ML_sampling import get_filtered_concept_ancestor,\
        generate_relation_positive_samples, \
        OffspringIterableDataset, AncestorIterableDataset
from modules.ML_data_condition_target_vars import *
from modules.timed_logger import logger

logger.reset_timer()
logger.log("Create positive samples for relation")
## Keep those concept relationship that exists in target_standard_ids
target_standard_ids = std_target['concept_id']
std_ids = conceptML['concept_id']
concept_ancestor_filtered = get_filtered_concept_ancestor(concept_ancestor, target_standard_ids, std_ids)
positive_dataset_relation = generate_relation_positive_samples(concept_ancestor_filtered, concept)
positive_dataset_relation = positive_dataset_relation[['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']]
len(positive_dataset_relation) # 376474





##########################
## Test offspring iterable dataset
##########################
n_neg=2
myiter1 = OffspringIterableDataset(
    positive_dataset_relation,
    std_target,
    n_neg=n_neg
)
len(myiter1)

it1 = iter(myiter1)
next(it1)
"""
{'sentence1': '[OFFSPRINT] Condition: X-linked dominant hereditary disease', 'sentence2': 'Condition: 2-methyl-3-hydroxybutyric aciduria', 'concept_id1': 37160643, 'concept_id2': 42538941, 'label': 1}
"""


##########################
## Test ancestor iterable dataset
##########################
myiter2 = AncestorIterableDataset(
    positive_dataset_relation,
    std_target,
    n_neg=n_neg
)
len(myiter2)

it2 = iter(myiter2)
next(it2)
"""
{'sentence1': 'Condition: X-linked dominant hereditary disease', 'sentence2': '<|parent of|> Condition: 2-methyl-3-hydroxybutyric aciduria', 'concept_id1': 37160643, 'concept_id2': 42538941, 'label': 1}
"""

logger.log("Save positive dataset for relation")
os.makedirs('data/ML/relation', exist_ok=True)
positive_dataset_relation.to_feather('data/ML/relation/positive_dataset_relation.feather')
std_target.to_feather('data/ML/relation/std_target.feather')

logger.done()