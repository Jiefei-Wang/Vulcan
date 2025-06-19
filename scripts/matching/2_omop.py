# Extract non-standard and synonym names from OMOP CDM
# File:
# - data/base_data/nonstd_names.feather
# - data/base_data/synonum_names.feather

import os
from tqdm import tqdm
import pandas as pd
from modules.timed_logger import logger
import duckdb

logger.reset_timer()

# Load OMOP CDM concept tables from feather files
concept = pd.read_feather('data/omop_feather/concept.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')
concept_synonym = pd.read_feather("data/omop_feather/concept_synonym.feather")
std_bridge = pd.read_feather("data/omop_feather/std_bridge.feather")

#######################################
## Standard to non-standard concept mapping
#######################################

logger.log("Extract a mapping of standard concepts to non-standard names")

# non-std to std mapping
name_map_OMOP_nonstd = std_bridge[std_bridge.concept_id!= std_bridge.std_concept_id]

## make sure two tables overlap with each other
map_table_OMOP_nonstd = concept.merge(
    name_map_OMOP_nonstd,
    on = 'concept_id',
    how = 'inner'
).rename(
    columns={
        'concept_id': 'source_id',
        'std_concept_id': 'concept_id',
        'concept_name': 'name'
        }
)


map_table_OMOP_nonstd['source'] = "OMOP"
map_table_OMOP_nonstd['type'] = 'nonstd'

map_table_OMOP_nonstd = map_table_OMOP_nonstd[['concept_id', 'source', 'source_id', 'type', 'name']]
# [3668243 rows x 5 columns]

#######################################
## Standard to synonym  mapping
#######################################
logger.log("Get concept synonyms from OMOP")

map_table_OMOP_synonyms = concept_synonym[['concept_id', 'concept_synonym_name']].rename(
    columns={
        'concept_synonym_name': 'name'
    }).drop_duplicates().reset_index(drop=True)


map_table_OMOP_synonyms['source_id'] = map_table_OMOP_synonyms['concept_id']
map_table_OMOP_synonyms['source'] = "OMOP"
map_table_OMOP_synonyms['type'] = 'synonym'
map_table_OMOP_synonyms = map_table_OMOP_synonyms[['concept_id', 'source', 'source_id', 'type', 'name']]
# [3240732 rows x 5 columns]

map_table_OMOP = pd.concat(
    [map_table_OMOP_nonstd, map_table_OMOP_synonyms],
    ignore_index=True
).drop_duplicates(subset=['concept_id', 'name']).reset_index(drop=True)


map_table_OMOP.to_feather('data/matching/map_table_OMOP.feather')
# [5186997 rows x 5 columns]

logger.done()