import pandas as pd
import os
from modules.timed_logger import logger
from sklearn.model_selection import train_test_split
from modules.CodeBlockExecutor import trace, tracedf

logger.reset_timer()
logger.log("Restricting to condition domain")

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

trace(target_concepts.shape)
#> (160288, 2)

# remove the concepts that do not have any mapping
mapped_concept_ids = matching_map_table['concept_id'].unique()
target_concepts = target_concepts[target_concepts['concept_id'].isin(mapped_concept_ids)].reset_index(drop=True)
target_concepts.to_feather(os.path.join(output_dir, 'target_concepts.feather'))

trace(target_concepts.shape)
#> (104198, 2)


# define the mapping table for condition domain
condition_matching_map_table = matching_map_table[matching_map_table['concept_id'].isin(target_concepts['concept_id'])].reset_index(drop=True)


tracedf(condition_matching_map_table)
#> DataFrame dimensions: 648285 rows × 6 columns
#> Column names:
#> ['concept_id', 'source', 'source_id', 'type', 'name_id', 'name']
#> Estimated memory usage: 185.05 MB


###################################
# Define the name bridge and name table for the entire dataset
###################################


condition_matching_name_bridge = condition_matching_map_table[['concept_id', 'name_id']]

condition_matching_name_table = condition_matching_map_table[['name_id', 'source', 'source_id', 'type', 'name']].drop_duplicates(subset=['name_id']).reset_index(drop=True)



tracedf(condition_matching_name_bridge)
#> DataFrame dimensions: 648285 rows × 2 columns
#> Column names:
#> ['concept_id', 'name_id']
#> Estimated memory usage: 9.89 MB

tracedf(condition_matching_name_table)
#> DataFrame dimensions: 648285 rows × 5 columns
#> Column names:
#> ['name_id', 'source', 'source_id', 'type', 'name']
#> Estimated memory usage: 180.10 MB


condition_matching_name_bridge.to_feather(os.path.join(output_dir, 'condition_matching_name_bridge.feather'))
condition_matching_name_table.to_feather(os.path.join(output_dir, 'condition_matching_name_table.feather'))



