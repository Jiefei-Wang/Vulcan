# Extract non-standard and synonym names from OMOP CDM
# File:
# - data/base_data/nonstd_names.feather
# - data/base_data/synonum_names.feather

import os
from tqdm import tqdm
import pandas as pd
from modules.timed_logger import logger
import duckdb
from modules.CodeBlockExecutor import tracedf

logger.reset_timer()
logger.log("Loading OMOP CDM concept tables")

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
OMOP_nonstd_std_pairs = concept.merge(
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


OMOP_nonstd_std_pairs['source'] = "OMOP"
OMOP_nonstd_std_pairs['type'] = 'nonstd'

OMOP_nonstd_std_pairs = OMOP_nonstd_std_pairs[['concept_id', 'source', 'source_id', 'type', 'name']]

trace(OMOP_nonstd_std_pairs.shape)
#> (3668243, 5)

#######################################
## Standard to synonym  mapping
#######################################
logger.log("Get concept synonyms from OMOP")

OMOP_synonyms_std_pairs = concept_synonym[['concept_id', 'concept_synonym_name']].rename(
    columns={
        'concept_synonym_name': 'name'
    }).drop_duplicates().reset_index(drop=True)


OMOP_synonyms_std_pairs['source_id'] = OMOP_synonyms_std_pairs['concept_id']
OMOP_synonyms_std_pairs['source'] = "OMOP"
OMOP_synonyms_std_pairs['type'] = 'synonym'
OMOP_synonyms_std_pairs = OMOP_synonyms_std_pairs[['concept_id', 'source', 'source_id', 'type', 'name']]

tracedf(OMOP_synonyms_std_pairs)
#> DataFrame dimensions: 4134188 rows × 5 columns
#> Column names:
#> ['concept_id', 'source', 'source_id', 'type', 'name']
#> Estimated memory usage: 964.03 MB

OMOP_pairs = pd.concat(
    [OMOP_nonstd_std_pairs, OMOP_synonyms_std_pairs],
    ignore_index=True
).drop_duplicates(subset=['concept_id', 'name']).reset_index(drop=True)


OMOP_pairs.to_feather('data/matching/OMOP_pairs.feather')
tracedf(OMOP_pairs)
#> DataFrame dimensions: 6384688 rows × 5 columns
#> Column names:
#> ['concept_id', 'source', 'source_id', 'type', 'name']
#> Estimated memory usage: 1.45 GB

logger.done()