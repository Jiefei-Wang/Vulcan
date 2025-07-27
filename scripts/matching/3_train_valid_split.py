import pandas as pd
import os
from modules.timed_logger import logger
from sklearn.model_selection import train_test_split
from modules.CodeBlockExecutor import trace, tracedf

logger.reset_timer()
logger.log("Combining all map_tables")

output_dir = "data/matching"
std_bridge = pd.read_feather("data/omop_feather/std_bridge.feather")
concept= pd.read_feather('data/omop_feather/concept.feather')
matching_map_table = pd.read_feather('data/matching/matching_map_table.feather')



####################
## Use the condition concepts for training
####################
logger.log("Define standard and non-standard concepts for training")
condition_concept = concept[concept['domain_id'] == 'Condition'].reset_index(drop=True)
std_condition_concept = condition_concept[condition_concept['standard_concept'] == 'S'].reset_index(drop=True)
nonstd_condition_concept = condition_concept[condition_concept['standard_concept'] != 'S'].reset_index(drop=True)


target_concepts = std_condition_concept[['concept_id', 'concept_name']].reset_index(drop=True)
target_concepts.to_feather(os.path.join(output_dir, 'target_concepts.feather'))

trace(target_concepts.shape)
#> (160288, 2)


# define the mapping table for condition domain
condition_matching_map_table = matching_map_table[matching_map_table['concept_id'].isin(target_concepts['concept_id'])].reset_index(drop=True)

tracedf(condition_matching_map_table)
#> DataFrame dimensions: 648285 rows × 5 columns
#> Column names:
#> ['concept_id', 'source', 'source_id', 'type', 'name']
#> Estimated memory usage: 180.10 MB

####################
## Exclude the CIEL concepts from map_table for validation purpose
## we will only exclude non-standard CIEL concept, so we will not lose any standard concepts
## In fact, there is no standard CIEL concept in the condition domain
####################
logger.log("Exclude the CIEL concepts from map_table")
reserved_vocab = "CIEL"

reserved_concepts = nonstd_condition_concept[nonstd_condition_concept.vocabulary_id == reserved_vocab]
reserved_ids = reserved_concepts.concept_id.astype(str)

# breaks the link between the reserved concepts and the standard concepts
row_filter_valid = (condition_matching_map_table.source == 'OMOP')&(condition_matching_map_table.source_id.isin(reserved_ids))

condition_matching_map_valid = condition_matching_map_table[row_filter_valid].reset_index(drop=True)
condition_matching_map_train = condition_matching_map_table[~row_filter_valid].reset_index(drop=True)

tracedf(condition_matching_map_train)
#> DataFrame dimensions: 643319 rows × 5 columns
#> Column names:
#> ['concept_id', 'source', 'source_id', 'type', 'name']
#> Estimated memory usage: 178.88 MB

tracedf(condition_matching_map_valid)
#> DataFrame dimensions: 4966 rows × 5 columns
#> Column names:
#> ['concept_id', 'source', 'source_id', 'type', 'name']
#> Estimated memory usage: 1.22 MB



# unique concept id in the table
trace(target_concepts['concept_id'].nunique())
#> 160288

# how many concepts have training data
trace(condition_matching_map_train['concept_id'].nunique())
#> 104198

# source of the training data
trace(condition_matching_map_train.groupby(['source', 'type'])['concept_id'].nunique())
#> source  type   
#> OMOP    nonstd     44669
#>         synonym    98677
#> UMLS    DEF        19553
#>         STR        24085
#> Name: concept_id, dtype: int64


####################
## Buld index for train data
## Some names are duplicated, so we need to create a unique id for each unique name
## two same names might be from different sources, however, we will only keep one of them
####################
logger.log(f"Buld index for train data")
unique_names = condition_matching_map_train['name'].unique()
name_to_id = pd.Series(range(1, len(unique_names) + 1), index=unique_names)
condition_matching_map_train['name_id'] = condition_matching_map_train['name'].map(name_to_id)

# matching_map_table['name_id'] = range(1, len(matching_map_table) + 1)

condition_matching_name_bridge_train = condition_matching_map_train[['concept_id', 'name_id']]
condition_matching_name_table_train = condition_matching_map_train[['name_id', 'source', 'source_id', 'type', 'name']].drop_duplicates(subset=['name_id']).reset_index(drop=True)


condition_matching_name_bridge_train.to_feather(os.path.join(output_dir, 'condition_matching_name_bridge_train.feather'))
condition_matching_name_table_train.to_feather(os.path.join(output_dir, 'condition_matching_name_table_train.feather'))

tracedf(condition_matching_name_bridge_train)
#> DataFrame dimensions: 643319 rows × 2 columns
#> Column names:
#> ['concept_id', 'name_id']
#> Estimated memory usage: 9.82 MB

tracedf(condition_matching_name_table_train)
#> DataFrame dimensions: 566536 rows × 5 columns
#> Column names:
#> ['name_id', 'source', 'source_id', 'type', 'name']
#> Estimated memory usage: 154.90 MB


####################
## Create valid and test mapping table
## sentence1(concept_name), sentence2 (name), concept_id1, concept_id2
####################
logger.log("Split valid and test data")

matching_valid_test = condition_matching_map_valid[['name', 'source', 'source_id', 'concept_id']].copy()
condition_matching_valid_test = matching_valid_test.merge(
    target_concepts,
    on = 'concept_id',
    how = 'inner'
).rename(
    columns={
        'concept_name': 'sentence1',
        'concept_id': 'concept_id1',
        'name': 'sentence2',
        'source_id': 'concept_id2'
    }
)

condition_matching_valid_test['label'] = 1  # Positive pairs
condition_matching_valid_test['concept_id2'] = condition_matching_valid_test['concept_id2'].astype('int64')
condition_matching_valid_test = condition_matching_valid_test[['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']].reset_index(drop=True)


condition_matching_valid_pos, condition_matching_test_pos = train_test_split(
    condition_matching_valid_test,
    test_size=0.9,
    random_state=42,
    shuffle=True)

condition_matching_valid_pos.reset_index(drop=True, inplace=True)
condition_matching_test_pos.reset_index(drop=True, inplace=True)


condition_matching_valid_pos.to_feather(os.path.join(output_dir, 'condition_matching_valid_pos.feather'))
condition_matching_test_pos.to_feather(os.path.join(output_dir, 'condition_matching_test_pos.feather'))

tracedf(condition_matching_valid_pos)
#> DataFrame dimensions: 496 rows × 5 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']
#> Estimated memory usage: 91.88 KB

tracedf(condition_matching_test_pos)
#> DataFrame dimensions: 4470 rows × 5 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']
#> Estimated memory usage: 835.24 KB







