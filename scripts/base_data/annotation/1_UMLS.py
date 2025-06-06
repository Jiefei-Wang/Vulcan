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

umls_starting_id = 10000*10000*10000

logger.log("Getting UMLS definitions")

concept = pd.read_feather('data/omop_feather/concept.feather')
std_bridge = pd.read_feather("data/omop_feather/std_bridge.feather")

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

## remove NAN from SAB
concept_mapped = concept_mapped.dropna(subset=['SAB'])

# Find the CUIs that are used in OMOP
mrconso_filtered = mrconso_df.merge(
    concept_mapped[['SAB', 'concept_code', 'concept_id']].drop_duplicates(),
    left_on=['SAB', 'CODE'],
    right_on=['SAB', 'concept_code'],
    how='inner'
)
mrconso_filtered['source_id'] = duckdb.query(f"""
    SELECT CUI || ':' || SAB || ':' || CODE AS generated_source_id
    FROM mrconso_filtered
""").df()['generated_source_id']


# Create concept to name mapping
# Concept_id -> SAB + concept_code -> source_id
concept_name_map_umls = mrconso_filtered[['source_id', 'concept_id']].drop_duplicates().reset_index(drop=True)


# create name_id to STR mapping
# Concept id -> SAB + concept_code -> STR
name_table_str = mrconso_filtered[['source_id', 'STR']].drop_duplicates().reset_index(drop=True)
name_table_str['source'] = 'UMLS'
name_table_str['type'] = 'STR'
name_table_str['name'] = name_table_str['STR']
name_table_str = name_table_str[['source_id', 'source', 'type', 'name']]


# create name_id to DEF mapping
# Concept_id -> SAB + concept_code -> CUI -> DEF
name_table_def = mrdef_df[['CUI', 'DEF']].drop_duplicates().reset_index(drop=True)
name_table_def = name_table_def.merge(
    mrconso_filtered[['CUI', 'source_id']],
    on='CUI',
    how='inner'
)



name_table_def['source'] = 'UMLS'
name_table_def['type'] = 'DEF'
name_table_def['name'] = name_table_def['DEF']
name_table_def = name_table_def.drop_duplicates(subset=['source_id', 'name']).reset_index(drop=True)
name_table_def = name_table_def[['source_id', 'source', 'type', 'name']]




## Check if all VOCAB_TO_SAB_MAP items have been used
vocab_keys = set(VOCAB_TO_SAB_MAP.values())
concept_vocabs = set(mrconso_filtered['SAB'].unique())
set_diff = (vocab_keys - concept_vocabs).union(concept_vocabs - vocab_keys)
if vocab_keys!=concept_vocabs:
    logger.warn(
        f"Unable to find definitions for vocabularies: {set_diff}. "
    )
    
concept_name_map_umls['name_id'] = concept_name_map_umls['source_id'].astype('category').cat.codes + umls_starting_id

name_table_umls = pd.concat([name_table_str, name_table_def], ignore_index=True)
name_table_umls = name_table_umls.merge(
    concept_name_map_umls[['source_id', 'name_id']],
    on='source_id',
    how='inner'
)

concept_name_map_umls = concept_name_map_umls[['source_id', 'name_id']]

concept_name_map_umls.to_feather('data/base_data/concept_name_map_umls.feather')
# [1926215 rows x 2 columns]
name_table_umls.to_feather('data/base_data/name_table_umls.feather')
# [4985807 rows x 5 columns]

