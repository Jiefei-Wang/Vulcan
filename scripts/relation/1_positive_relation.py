import pandas as pd
import os
from modules.TOKENS import TOKENS
from sklearn.model_selection import train_test_split
from modules.timed_logger import logger
from modules.CodeBlockExecutor import trace, tracedf
from modules.TOKENS import TOKENS

logger.reset_timer()
logger.log("creating relation pairs")


omop_base_path = "data/omop_feather"
matching_base_path = "data/matching"
relation_base_path = "data/relation"

if not os.path.exists(relation_base_path):
    os.makedirs(relation_base_path)

target_concepts = pd.read_feather(os.path.join(matching_base_path, 'target_concepts.feather'))
concept_ancestors = pd.read_feather(os.path.join(omop_base_path, 'concept_ancestor.feather'))
concept = pd.read_feather(os.path.join(omop_base_path, 'concept.feather'))

# We want both ancestor and descendant to be in target_concepts
relation_tables = concept_ancestors[
    concept_ancestors['ancestor_concept_id'].isin(target_concepts['concept_id'])&
    (concept_ancestors['min_levels_of_separation'] >= 1) &
    (concept_ancestors['min_levels_of_separation'] <= 2)
    ].drop_duplicates(subset=['ancestor_concept_id', 'descendant_concept_id']).reset_index(drop=True)


name_bridge_relation = relation_tables.rename(
    columns={
        'ancestor_concept_id': 'concept_id',
        'descendant_concept_id': 'name_id'
    }
    )[[ 'concept_id', 'name_id']].reset_index(drop=True)


relation_pos = name_bridge_relation.merge(
    concept[['concept_id', 'concept_name']].rename(columns={'concept_id': 'concept_id', 'concept_name': 'sentence1'}),
    on='concept_id', how='left'
).merge(
    concept[['concept_id', 'concept_name']].rename(columns={'concept_id': 'name_id', 'concept_name': 'sentence2'}),
    on='name_id', how='left'
    )


relation_pos = relation_pos[['sentence1', 'sentence2', 'concept_id', 'name_id']].reset_index(drop=True)

relation_pos['sentence2'] = relation_pos['sentence2'].apply(lambda x: TOKENS.parent + x)

relation_pos.to_feather(os.path.join(relation_base_path, 'relation_positive.feather'))


tracedf(relation_pos)
#> DataFrame dimensions: 1038070 rows × 4 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id', 'name_id']
#> Estimated memory usage: 200.87 MB



####################
## Train, test, valid split
####################
logger.log("Train, test, valid split")

test_num = 5000
valid_num = 5000

# Randomly select unique concept IDs for test and validation
unique_concept_ids = relation_pos['concept_id'].unique()
trace(len(unique_concept_ids))
#> 42058

remaining_concept_ids, test_concept_ids = train_test_split(
    unique_concept_ids, test_size=test_num, random_state=42, shuffle=True
)
train_concept_ids, valid_concept_ids = train_test_split(
    remaining_concept_ids, test_size=valid_num, random_state=42, shuffle=True
)

# For each concept ID, randomly select one pair
test_indices = relation_pos[relation_pos['concept_id'].isin(test_concept_ids)].groupby('concept_id').sample(n=1, random_state=42).index.tolist()
valid_indices = relation_pos[relation_pos['concept_id'].isin(valid_concept_ids)].groupby('concept_id').sample(n=1, random_state=42).index.tolist()
train_indices = list(set(relation_pos.index) - set(test_indices) - set(valid_indices))


assert len(train_indices) + len(valid_indices) + len(test_indices) == len(relation_pos)


relation_pos_train = relation_pos.loc[train_indices].reset_index(drop=True)
relation_pos_valid = relation_pos.loc[valid_indices].reset_index(drop=True)
relation_pos_test = relation_pos.loc[test_indices].reset_index(drop=True)

os.makedirs(os.path.join(relation_base_path, 'train'), exist_ok=True)
relation_pos_train.to_feather(os.path.join(relation_base_path, 'train/relation_pos_train.feather'))
relation_pos_valid.to_feather(os.path.join(relation_base_path, 'train/relation_pos_valid.feather'))
relation_pos_test.to_feather(os.path.join(relation_base_path, 'train/relation_pos_test.feather'))


tracedf(relation_pos_train)
#> DataFrame dimensions: 1028070 rows × 4 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id', 'name_id']
#> Estimated memory usage: 198.96 MB

tracedf(relation_pos_valid)
#> DataFrame dimensions: 5000 rows × 4 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id', 'name_id']
#> Estimated memory usage: 976.14 KB

tracedf(relation_pos_test)
#> DataFrame dimensions: 5000 rows × 4 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id', 'name_id']
#> Estimated memory usage: 975.85 KB
