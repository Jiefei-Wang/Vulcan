# Create a concept_id to UMLS description table: concept_id_to_umls
# columns=['concept_id', 'umls'] where umls is a list of UMLS definitions
# Files:
# - data/base_data/umls_names.feather
# - data/base_data/concept_id_to_umls.feather
import os
from tqdm import tqdm
import pandas as pd
from modules.timed_logger import logger
import duckdb
logger.reset_timer()

logger.log("Getting UMLS definitions")

concept = pd.read_feather('data/omop_feather/concept.feather')

# Load UMLS reference files
mrconso_df = pd.read_feather("data/UMLS_feather/mrconso_df.feather")
mrdef_df = pd.read_feather("data/UMLS_feather/mrdef_df.feather")

# Mapping from OMOP vocabulary_id to UMLS SAB 
# TODO: Verify if these mappings are correct and complete
# OMOP: https://athena.ohdsi.org/vocabulary/list
# UMLS: https://www.nlm.nih.gov/research/umls/sourcereleasedocs/
VOCAB_TO_SAB_MAP = {
    "SNOMED": "SNOMEDCT_US",  # OMOP: Systematic Nomenclature of Medicine - Clinical Terms -> UMLS: SNOMED CT, US Edition
    "ICD9CM": "ICD9CM",      # OMOP: International Classification of Diseases, Ninth Revision, Clinical Modification... -> UMLS: International Classification of Diseases, Ninth Revision, Clinical Modification
    "CPT4": "CPT",          # OMOP: Current Procedural Terminology version 4 -> UMLS: CPT - Current Procedural Terminology
    "HCPCS": "HCPCS",       # OMOP: Healthcare Common Procedure Coding System -> UMLS: HCPCS - Healthcare Common Procedure Coding System
    "LOINC": "LNC",         # OMOP: Logical Observation Identifiers Names and Codes -> UMLS: LOINC
    "RxNorm": "RXNORM",     # OMOP: RxNorm -> UMLS: RXNORM
    "MedDRA": "MDR",        # OMOP: Medical Dictionary for Regulatory Activities -> UMLS: MedDRA
    "Read": "RCD",          # OMOP: NHS UK Read Codes Version 2 -> UMLS: Read Codes
    "ATC": "ATC",           # OMOP: WHO Anatomic Therapeutic Chemical Classification -> UMLS: Anatomical Therapeutic Chemical Classification System
    "VANDF": "VANDF",       # OMOP: Veterans Health Administration National Drug File (VA) -> UMLS: National Drug File
    "ICD10": "ICD10",       # OMOP: International Classification of Diseases, Tenth Revision -> UMLS: International Classification of Diseases and Related Health Problems, Tenth Revision
    "ICD10PCS": "ICD10PCS", # OMOP: ICD-10 Procedure Coding System -> UMLS: ICD-10 Procedure Coding System
    "MeSH": "MSH",          # OMOP: Medical Subject Headings -> UMLS: MeSH
    "NUCC": "NUCCHCPT",     # OMOP: National Uniform Claim Committee Health Care Provider Taxonomy Code Set -> UMLS: National Uniform Claim Committee - Health Care Provider Taxonomy
    "SPL": "MTHSPL",        # OMOP: Structured Product Labeling -> UMLS: FDA Structured Product Labels
    "CVX": "CVX",           # OMOP: CDC Vaccine Administered CVX -> UMLS: Vaccines Administered
    "ICD10CM": "ICD10CM",   # OMOP: International Classification of Diseases, Tenth Revision, Clinical Modification -> UMLS: International Classification of Diseases, Tenth Revision, Clinical Modification
    "CDT": "CDT",           # OMOP: Current Dental Terminology -> UMLS: CDT
    "MEDRT": "MED-RT",      # OMOP: Medication Reference Terminology MED-RT (VA) -> UMLS: Medication Reference Terminology
    "SNOMED Veterinary": "SNOMEDCT_VET", # OMOP: SNOMED Veterinary Extension -> UMLS: SNOMED CT, Veterinary Extension
    "HGNC": "HGNC",         # OMOP: Human Gene Nomenclature -> UMLS: HUGO Gene Nomenclature Committee
    "NCIt": "NCI"           # OMOP: NCI Thesaurus -> UMLS: NCI Thesaurus
}

concept_mapped = concept.copy()
concept_mapped['SAB'] = concept_mapped['vocabulary_id'].map(VOCAB_TO_SAB_MAP)

## remove NAN from SAB (not mapped vocabularies)
concept_mapped = concept_mapped.dropna(subset=['SAB'])

# Find the CUIs that are used in OMOP and add its concept_id
mrconso_filtered = duckdb.query(f"""
    SELECT DISTINCT
    mrconso.CUI,
    concept.concept_id,
    mrconso.STR,
    mrconso.SAB,
    mrconso.CUI || ':' || mrconso.SAB || ':' || mrconso.CODE AS source_id
    FROM mrconso_df as mrconso
    JOIN concept_mapped as concept
    ON mrconso.SAB = concept.SAB AND mrconso.CODE = concept.concept_code
""").df()



## Check if all VOCAB_TO_SAB_MAP items have been used
vocab_keys = set(VOCAB_TO_SAB_MAP.values())
concept_vocabs = set(mrconso_filtered['SAB'].unique())
set_diff = (vocab_keys - concept_vocabs).union(concept_vocabs - vocab_keys)
if vocab_keys!=concept_vocabs:
    logger.warn(
        f"Unable to find definitions for vocabularies: {set_diff}. "
    )
    

####################
## UMLS STR names
####################
map_table_umls_str = mrconso_filtered[['source_id', 'concept_id','STR']].rename(
    columns={
        'STR': 'name'
    }
).reset_index(drop=True)
map_table_umls_str['source'] = 'UMLS'
map_table_umls_str['type'] = 'STR'
map_table_umls_str=map_table_umls_str[['concept_id', 'source', 'source_id', 'type', 'name']]

####################
## UMLS DEF names
####################
# Concept_id -> SAB + concept_code -> CUI
concept_id_CUI_bridge = mrconso_filtered[['CUI', 'concept_id']].drop_duplicates().reset_index(drop=True)

# create name_id to DEF mapping
# Concept_id -> CUI -> DEF
map_table_umls_def = mrdef_df[['CUI', 'DEF']].drop_duplicates().reset_index(drop=True)
map_table_umls_def = map_table_umls_def.merge(
    concept_id_CUI_bridge,
    on='CUI',
    how='inner'
).rename(
    columns={
        'DEF': 'name'
    }
)

map_table_umls_def['source_id'] = map_table_umls_def['CUI']  
map_table_umls_def['source'] = 'UMLS'
map_table_umls_def['type'] = 'DEF'
map_table_umls_def = map_table_umls_def[['concept_id', 'source', 'source_id', 'type', 'name']]



map_table_umls = pd.concat([map_table_umls_str, map_table_umls_def], ignore_index=True).drop_duplicates(subset=['concept_id', 'name']).reset_index(drop=True)
    

map_table_umls.to_feather('data/matching/map_table_umls.feather')
# [4985537 rows x 5 columns]
logger.done()