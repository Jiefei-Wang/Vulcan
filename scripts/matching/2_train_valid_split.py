import pandas as pd
import os
from modules.timed_logger import logger
from sklearn.model_selection import train_test_split

logger.reset_timer()
logger.log("Combining all map_tables")

std_bridge = pd.read_feather("data/omop_feather/std_bridge.feather")
concept= pd.read_feather('data/omop_feather/concept.feather')

base_path = "data/matching"
map_tables = [
    'map_table_umls.feather',
    'map_table_OMOP.feather',
]

matching_map_table = pd.concat(
    [pd.read_feather(os.path.join(base_path, name_map_table)) for name_map_table in map_tables],
    ignore_index=True
)
matching_map_table['source_id'] = matching_map_table['source_id'].astype(str)

# keep only the concepts that are in the standard bridge
#   1. if the concept_id is standard, it is in the std_bridge
#   2. if the concept_id is non-standard, it is in the std_bridge
#      if it has a mapping to a standard concept
# For thos that are not in the std_bridge, there is no way to map them 
# to a standard concept, so we will not use them in the training.
matching_map_table = matching_map_table.merge(
    std_bridge[['concept_id', 'std_concept_id']],
    on ='concept_id',
    how='inner'
).drop(columns=['concept_id']).rename(
    columns={
        'std_concept_id': 'concept_id' # We will use the standard concept_id
    }
).reset_index(drop=True)
matching_map_table = matching_map_table[['concept_id', 'source', 'source_id', 'type', 'name']]

matching_map_table.to_feather(os.path.join(base_path, 'matching_map_table.feather'))


logger.log("Define standard and non-standard concepts")
condition_concept = concept[concept['domain_id'] == 'Condition'].reset_index(drop=True)
std_condition_concept = condition_concept[condition_concept['standard_concept'] == 'S'].reset_index(drop=True)
nonstd_condition_concept = condition_concept[condition_concept['standard_concept'] != 'S'].reset_index(drop=True)


std_condition_concept.to_feather(os.path.join(base_path, 'std_condition_concept.feather'))

# define the mapping table for condition domain
condition_matching_map_table = matching_map_table[matching_map_table['concept_id'].isin(std_condition_concept['concept_id'])].reset_index(drop=True)

####################
## Exclude the reserved concepts from map_table
####################
logger.log("Exclude the reserved concepts from map_table")
reserved_vocab = "CIEL"

reserved_concepts = nonstd_condition_concept[nonstd_condition_concept.vocabulary_id == reserved_vocab]
reserved_ids = reserved_concepts.concept_id.astype(str)

# breaks the link between the reserved concepts and the standard concepts
row_filter_valid = (condition_matching_map_table.source == 'OMOP')&(condition_matching_map_table.source_id.isin(reserved_ids))

condition_matching_map_valid = condition_matching_map_table[row_filter_valid].reset_index(drop=True)
condition_matching_map_train = condition_matching_map_table[~row_filter_valid].reset_index(drop=True)


####################
## Buld index for train data
####################
logger.log(f"Buld index for train data")
unique_names = condition_matching_map_train['name'].unique()
name_to_id = pd.Series(range(1, len(unique_names) + 1), index=unique_names)
condition_matching_map_train['name_id'] = condition_matching_map_train['name'].map(name_to_id)

# matching_map_table['name_id'] = range(1, len(matching_map_table) + 1)

condition_matching_name_bridge_train = condition_matching_map_train[['concept_id', 'name_id']]
condition_matching_name_table_train = condition_matching_map_train[['name_id', 'source', 'source_id', 'type', 'name']].drop_duplicates(subset=['name_id']).reset_index(drop=True)


condition_matching_name_bridge_train.to_feather(os.path.join(base_path, 'condition_matching_name_bridge_train.feather'))
# [1624671 rows x 2 columns]
condition_matching_name_table_train.to_feather(os.path.join(base_path, 'condition_matching_name_table_train.feather'))
# [764392 rows x 5 columns]



####################
## Create valid and test mapping table
## sentence1, sentence2, concept_id1, concept_id2
####################
matching_valid_test = condition_matching_map_valid[['name', 'source', 'source_id', 'concept_id']].copy()
condition_matching_valid_test = matching_valid_test.merge(
    std_condition_concept[['concept_id', 'concept_name']],
    on = 'concept_id',
    how = 'inner'
).rename(
    columns={
        'name': 'sentence1',
        'source_id': 'concept_id1',
        'concept_name': 'sentence2',
        'concept_id': 'concept_id2'
    }
)

condition_matching_valid_test = condition_matching_valid_test[['sentence1', 'sentence2', 'concept_id1', 'concept_id2']].reset_index(drop=True)


## split the valid and test data
logger.log("Split valid and test data")

condition_matching_valid, condition_matching_test = train_test_split(
    condition_matching_valid_test,
    test_size=0.9,
    random_state=42,
    shuffle=True)

condition_matching_valid.reset_index(drop=True, inplace=True)
condition_matching_test.reset_index(drop=True, inplace=True)


condition_matching_valid.to_feather(os.path.join(base_path, 'condition_matching_valid.feather'))

condition_matching_test.to_feather(os.path.join(base_path, 'condition_matching_test.feather'))
# [30737 rows x 4 columns]
logger.done()

