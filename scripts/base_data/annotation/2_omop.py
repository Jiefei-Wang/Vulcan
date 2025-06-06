# Extract non-standard and synonym names from OMOP CDM
# File:
# - data/base_data/nonstd_names.feather
# - data/base_data/synonum_names.feather

import os
from tqdm import tqdm
import pandas as pd
from modules.timed_logger import logger
import duckdb


# Load OMOP CDM concept tables from feather files
concept = pd.read_feather('data/omop_feather/concept.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')
concept_synonym = pd.read_feather("data/omop_feather/concept_synonym.feather")
std_bridge = pd.read_feather("data/omop_feather/std_bridge.feather")

#######################################
## Standard to non-standard concept mapping
#######################################
logger.log("Extract a mapping of standard concepts to non-standard names")


concept_name_map_nonstd = concept_relationship[concept_relationship['relationship_id'] == 'Maps to']
concept_name_map_nonstd = concept_name_map_nonstd[['concept_id_1', 'concept_id_2']].rename(
    columns={
        'concept_id_1': 'source_id',
        'concept_id_2': 'concept_id'
    }
)
concept_name_map_nonstd = concept_name_map_nonstd[concept_name_map_nonstd.source_id!= concept_name_map_nonstd.concept_id].reset_index(drop=True)

nonstd_concepts = concept[concept['standard_concept'] != 'S']
name_table_nonstd = nonstd_concepts[['concept_id', 'concept_name']].rename(
    columns={'concept_name': 'name'}
    )

name_table_nonstd['source_id'] = name_table_nonstd['concept_id']
name_table_nonstd['source'] = "OMOP"
name_table_nonstd['type'] = 'nonstd'


name_table_nonstd.to_feather('data/base_data/name_table_nonstd.feather')


#######################################
## Standard to synonym  mapping
#######################################
logger.log("Get concept synonyms from OMOP")


synonum_names = concept_synonym[['concept_id', 'concept_synonym_name']].rename(
    columns={
        'concept_synonym_name': 'source_name'
    })

synonum_names['source_id'] = synonum_names['concept_id']
synonum_names['type'] = 'synonym'
synonum_names['source'] = "OMOP"
synonum_names.to_feather('data/base_data/synonum_names.feather')