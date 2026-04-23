import pandas as pd
import os
from modules.timed_logger import logger
from modules.CodeBlockExecutor import trace, tracedf

logger.reset_timer()
logger.log("Restricting to condition domain")

output_dir = "data/matching"
std_bridge = pd.read_feather("data/omop_feather/std_bridge.feather")
concept= pd.read_feather('data/omop_feather/concept.feather')
matching_pairs = pd.read_feather('data/matching/matching_pairs.feather')



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
mapped_concept_ids = matching_pairs['concept_id'].unique()
target_concepts = target_concepts[target_concepts['concept_id'].isin(mapped_concept_ids)].reset_index(drop=True)
target_concepts.to_feather(os.path.join(output_dir, 'target_concepts.feather'))

trace(target_concepts.shape)
#> (101118, 2)


# define the mapping table for condition domain
condition_matching_pairs = matching_pairs[matching_pairs['concept_id'].isin(target_concepts['concept_id'])].reset_index(drop=True)


tracedf(condition_matching_pairs)
#> DataFrame dimensions: 545158 rows × 7 columns
#> Column names:
#> ['concept_id', 'concept_name', 'source', 'source_id', 'type', 'name_id', 'name']
#> Estimated memory usage: 208.91 MB

condition_matching_pairs.to_feather(os.path.join(output_dir, 'condition_matching_pairs.feather'))