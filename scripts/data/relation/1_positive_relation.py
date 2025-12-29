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
relation_train_dir = os.path.join(relation_base_path, 'train')

if not os.path.exists(relation_train_dir):
    os.makedirs(relation_train_dir)

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


relation_pairs = name_bridge_relation.merge(
    concept[['concept_id', 'concept_name']],
    on='concept_id', how='left'
).merge(
    concept[['concept_id', 'concept_name']].rename(columns={'concept_id': 'name_id', 'concept_name': 'name'}),
    on='name_id', how='left'
    )


relation_pairs = relation_pairs[['concept_name', 'name', 'concept_id', 'name_id']].reset_index(drop=True)

relation_pairs['name'] = relation_pairs['name'].apply(lambda x: TOKENS.parent + x)

relation_pairs.to_feather(os.path.join(relation_base_path, 'relation_pairs.feather'))


tracedf(relation_pairs)
#> DataFrame dimensions: 1036046 rows × 4 columns
#> Column names:
#> ['concept_name', 'name', 'concept_id', 'name_id']
#> Estimated memory usage: 216.34 MB

relation_pairs.to_feather(os.path.join(relation_base_path, 'relation_pairs.feather'))


####################
## Train, test, valid split
####################
logger.log("Train, test, valid split")

test_num = 5000
valid_num = 5000


indices = relation_pairs.index.tolist()
train_idx, temp_idx = train_test_split(
    indices, test_size=test_num + valid_num, random_state=42, shuffle=True
)
valid_idx, test_idx = train_test_split(
    temp_idx, test_size=test_num, random_state=42, shuffle=True
)

relation_pairs_train = relation_pairs.loc[train_idx].reset_index(drop=True)
relation_pairs_valid = relation_pairs.loc[valid_idx].reset_index(drop=True)
relation_pairs_test = relation_pairs.loc[test_idx].reset_index(drop=True)

relation_pairs_train.to_feather(os.path.join(relation_train_dir, 'relation_pairs_train.feather'))
relation_pairs_valid.to_feather(os.path.join(relation_train_dir, 'relation_pairs_valid.feather'))
relation_pairs_test.to_feather(os.path.join(relation_train_dir, 'relation_pairs_test.feather'))


tracedf(relation_pairs_train)
#> DataFrame dimensions: 1026046 rows × 4 columns
#> Column names:
#> ['concept_name', 'name', 'concept_id', 'name_id']
#> Estimated memory usage: 214.25 MB

tracedf(relation_pairs_valid)
#> DataFrame dimensions: 5000 rows × 4 columns
#> Column names:
#> ['concept_name', 'name', 'concept_id', 'name_id']
#> Estimated memory usage: 1.04 MB

tracedf(relation_pairs_test)
#> DataFrame dimensions: 5000 rows × 4 columns
#> Column names:
#> ['concept_name', 'name', 'concept_id', 'name_id']
#> Estimated memory usage: 1.04 MB
