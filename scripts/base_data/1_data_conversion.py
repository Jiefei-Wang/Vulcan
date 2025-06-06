# This script 
#   - convert the OMOP concept from csv to feather format
#   - conceptEX: add a column 'std_concept_id_list'
#       - keeps all concepts, setting std_concept_id_list to [] for unmapped
#   - Create a conceptEX table that contains std_concept_id_list column
#   - convert the UMLS concept from csv to feather format
import pandas as pd
import numpy as np
import duckdb
import os
from modules.timed_logger import logger
logger.reset_timer()


#############################
## OMOP data
#############################


logger.log("Loading OMOP concept tables")
# Read dataset
omop_root = 'data/omop_csv'
omop_clean_root = 'data/omop_feather'
concept = pd.read_csv(os.path.join(omop_root, 'CONCEPT.csv'), delimiter='\t', low_memory=False, na_filter=False)
concept_relationship = pd.read_csv(os.path.join(omop_root, 'CONCEPT_RELATIONSHIP.csv'), delimiter='\t', low_memory=False, na_filter=False)
concept_synonym = pd.read_csv(os.path.join(omop_root, 'CONCEPT_SYNONYM.csv'), delimiter='\t', low_memory=False, na_filter=False)
concept_ancestor = pd.read_csv(os.path.join(omop_root, 'CONCEPT_ANCESTOR.csv'), delimiter='\t', low_memory=False, na_filter=False)

concept.to_feather(os.path.join(omop_clean_root, 'concept.feather'))
concept_relationship.to_feather(os.path.join(omop_clean_root, 'concept_relationship.feather'))
concept_synonym.to_feather(os.path.join(omop_clean_root, 'concept_synonym.feather'))
concept_ancestor.to_feather(os.path.join(omop_clean_root, 'concept_ancestor.feather'))


std_concepts = concept[concept['standard_concept'] == 'S']
nonstd_concepts = concept[concept['standard_concept'] != 'S']
len(concept) # 9683303
len(std_concepts) # 3490518
len(nonstd_concepts) # 6192785

# nonstd to std mapping
std_bridge = concept_relationship[concept_relationship.relationship_id=='Maps to'][['concept_id_1', 'concept_id_2']].rename(
    columns={
        'concept_id_1': 'concept_id',
        'concept_id_2': 'std_concept_id'
        }
    )


## Combine the standard and non-standard concepts
std_bridge.to_feather(os.path.join(omop_clean_root, 'std_bridge.feather'))



#############################
## UMLS data
#############################
logger.log("Loading UMLS tables")
def read_mrconso(mrconso_path):
    mrconso_columns = [
        "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI",
        "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"
    ]
    mrconso_df = pd.read_csv(
        mrconso_path, delimiter="|", names=mrconso_columns, dtype=str, header=None, index_col=False
    )
    # Drop the last empty column caused by the trailing delimiter
    mrconso_df = mrconso_df.drop(columns=[mrconso_df.columns[-1]])
    return mrconso_df


# Load UMLS MRDEF.RRF (definitions)
def read_mrdef(umls_def_path):
    umls_def_columns = ["CUI", "AUI", "ATUI", "SATUI", "SAB", "DEF", "SUPPRESS", "CVF"]
    UMLS_def = pd.read_csv(
        umls_def_path, delimiter="|", names=umls_def_columns, dtype=str, header=None, index_col=False
    )
    # Drop the last empty column
    UMLS_def = UMLS_def.drop(columns=[UMLS_def.columns[-1]])
    return UMLS_def


umls_root = 'data/UMLS_raw'
umls_clean_root = 'data/UMLS_feather'

mrconso_path = os.path.join(umls_root, 'MRCONSO.RRF')
umls_def_path = os.path.join(umls_root, 'MRDEF.RRF')
mrconso = read_mrconso(mrconso_path) # Reads UMLS concept names and relationships
mrdef_df = read_mrdef(umls_def_path) # Reads UMLS concept definitions

mrconso_df = mrconso[['CUI', 'SAB', 'CODE', 'STR']]


## create if not exist
if not os.path.exists(umls_clean_root):
    os.makedirs(umls_clean_root)

mrconso_df.to_feather(os.path.join(umls_clean_root, 'mrconso_df.feather'))
mrdef_df.to_feather(os.path.join(umls_clean_root, 'mrdef_df.feather'))

logger.done()