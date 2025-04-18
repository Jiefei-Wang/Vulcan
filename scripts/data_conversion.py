## This script convert the OMOP concept from csv to feather format
## Key modifications:
##  1. Add a column 'std_concept_id' to the concept table.
##  2. Remove the concept with no mapping to standard concept.
import pandas as pd
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

## merge nonstd_conditions with concept_relationship to get the standard_concept_id
std_map = concept_relationship[concept_relationship.relationship_id=='Maps to'][['concept_id_1', 'concept_id_2']].rename(columns={'concept_id_2': 'std_concept_id'})

## map, for multple mappings, combine them into a list
conceptEX = concept.merge(
    std_map, left_on='concept_id', right_on='concept_id_1', 
    how='left'
    ).drop(columns=['concept_id_1'])

conceptEX['std_concept_id'] = conceptEX['std_concept_id'].astype('Int64')


## group by concept_id and combine the standard_concept_id into a list
conceptEX = conceptEX.groupby('concept_id'
    ).agg({
        'concept_name': 'first',
        'domain_id': 'first',
        'vocabulary_id': 'first',
        'concept_class_id': 'first',
        'standard_concept': 'first',
        'concept_code': 'first',
        'valid_start_date': 'first',
        'valid_end_date': 'first',
        'invalid_reason': 'first',
        'std_concept_id': lambda x: list(x)}
    ).reset_index()


## Combine the standard and non-standard concepts
conceptEX.to_feather(os.path.join(omop_clean_root, 'conceptEX.feather'))



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

mrconso_df = mrconso[['CUI', 'SAB', 'CODE']]

## create if not exist
if not os.path.exists(umls_clean_root):
    os.makedirs(umls_clean_root)

mrconso_df.to_feather(os.path.join(umls_clean_root, 'mrconso_df.feather'))
mrdef_df.to_feather(os.path.join(umls_clean_root, 'mrdef_df.feather'))

logger.done()