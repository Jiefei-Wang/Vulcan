import pandas as pd
import os
from modules.timed_logger import logger
from sklearn.model_selection import train_test_split
from modules.CodeBlockExecutor import trace, tracedf

logger.reset_timer()
logger.log("Combining all map_tables")

output_dir = "data/matching"
output_train_dir = os.path.join(output_dir, 'train')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_train_dir):
    os.makedirs(output_train_dir)


# std_bridge = pd.read_feather("data/omop_feather/std_bridge.feather")
concept= pd.read_feather('data/omop_feather/concept.feather')
# matching_map_table = pd.read_feather('data/matching/matching_map_table.feather')


condition_matching_name_bridge = pd.read_feather('data/matching/condition_matching_name_bridge.feather')
condition_matching_name_table = pd.read_feather('data/matching/condition_matching_name_table.feather')



####################
## Train, test, valid split
####################
logger.log("Train, test, valid split")

test_num = 5000
valid_num = 5000

# Randomly select unique concept IDs for test and validation
unique_concept_ids = condition_matching_name_bridge['concept_id'].unique()
trace(len(unique_concept_ids))
#> 104198

remaining_concept_ids, test_concept_ids = train_test_split(
    unique_concept_ids, test_size=test_num, random_state=42, shuffle=True
)
train_concept_ids, valid_concept_ids = train_test_split(
    remaining_concept_ids, test_size=valid_num, random_state=42, shuffle=True
)

# For each concept ID, randomly select one pair
test_indices = condition_matching_name_bridge[condition_matching_name_bridge['concept_id'].isin(test_concept_ids)].groupby('concept_id').sample(n=1, random_state=42).index.tolist()
valid_indices = condition_matching_name_bridge[condition_matching_name_bridge['concept_id'].isin(valid_concept_ids)].groupby('concept_id').sample(n=1, random_state=42).index.tolist()
train_indices = list(set(condition_matching_name_bridge.index) - set(test_indices) - set(valid_indices))


assert len(train_indices) + len(valid_indices) + len(test_indices) == len(condition_matching_name_bridge)


# sentence1 is always standard concept name
# sentence2 is the non-standard name
def create_train_df(name_bridge, name_table, concept):
    name_id_2_name = name_table[['name_id', "name"]].copy()
    concept_id_2_name = concept[['concept_id', 'concept_name']]
    df = name_bridge.merge(
        concept_id_2_name,
        on = 'concept_id',
        how = 'inner'
    ).merge(
        name_id_2_name,
        on = 'name_id',
        how = 'inner'
    ).rename(
        columns={
            'concept_name': 'sentence1',
            'name': 'sentence2'
        }
    )
    return df[['sentence1', 'sentence2', 'concept_id', 'name_id']].reset_index(drop=True)



train_name_bridge = condition_matching_name_bridge.loc[train_indices].reset_index(drop=True)
valid_name_bridge = condition_matching_name_bridge.loc[valid_indices].reset_index(drop=True)
test_name_bridge = condition_matching_name_bridge.loc[test_indices].reset_index(drop=True)

matching_pos = create_train_df(condition_matching_name_bridge, condition_matching_name_table, concept)


matching_pos_train = create_train_df(train_name_bridge, condition_matching_name_table, concept)
matching_pos_valid = create_train_df(valid_name_bridge, condition_matching_name_table, concept)
matching_pos_test = create_train_df(test_name_bridge, condition_matching_name_table, concept)


matching_pos.to_feather(os.path.join(output_dir, 'matching_pos.feather'))
matching_pos_train.to_feather(os.path.join(output_train_dir, 'matching_pos_train.feather'))
matching_pos_valid.to_feather(os.path.join(output_train_dir, 'matching_pos_valid.feather'))
matching_pos_test.to_feather(os.path.join(output_train_dir, 'matching_pos_test.feather'))

tracedf(matching_pos)
#> DataFrame dimensions: 648285 rows × 4 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id', 'name_id']
#> Estimated memory usage: 132.71 MB

tracedf(matching_pos_train)
#> DataFrame dimensions: 638285 rows × 4 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id', 'name_id']
#> Estimated memory usage: 130.72 MB

tracedf(matching_pos_valid)
#> DataFrame dimensions: 5000 rows × 4 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id', 'name_id']
#> Estimated memory usage: 1,018.20 KB

tracedf(matching_pos_test)
#> DataFrame dimensions: 5000 rows × 4 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id', 'name_id']
#> Estimated memory usage: 1,021.35 KB