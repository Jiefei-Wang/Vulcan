# This file is used to generate the mapping from 
# standard to non-standard concepts
#
# For each standard concept, create columns:
# - nonstd_ids: list of non-standard concept ids
# - nonstd_names: list of non-standard concept names (size = nonstd_id)
# - descriptions: list of concept descriptions (from UMLS)
# - synonym_names: list of concept synonyms (if no synonym, then empty list)
# If a value in a column is empty, it must be None
# 
# Store the result in data/ML/conceptML.feather
import importlib
import modules
from tqdm import tqdm
import pandas as pd
from modules.ML_extract_name import extract_nonstd_names, extract_synonym, extract_umls_description
from modules.UMLS_file import read_mrconso, read_mrdef

# Reload the custom module to ensure latest version of functions are used
importlib.reload(modules.ML_extract_name)


# Load OMOP CDM concept tables from feather files
concept = pd.read_feather('data/omop_feather/concept.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')
concept_synonym = pd.read_feather("data/omop_feather/concept_synonym.feather")


# Load and process UMLS reference files
# UMLS data contains descriptions of the concepts
mrconso_path = "data/UMLS_raw/MRCONSO.RRF"
umls_def_path = "data/UMLS_raw/MRDEF.RRF"
mrconso_df = read_mrconso(mrconso_path) # Reads UMLS concept names and relationships
mrdef_df = read_mrdef(umls_def_path) # Reads UMLS concept definitions


# Extract standard concept names and descriptions
std_concept = concept[concept['standard_concept'] == 'S']
nonstd_concept = concept[concept['standard_concept'] != 'S']


std_conditions = std_concept[std_concept.domain_id == 'Condition']
std_conditions.groupby('vocabulary_id')['concept_id'].count()
"""
HCPCS                    1
ICDO3                56858
Nebraska Lexicon      1274
OMOP Extension         341
OPCS4                    1
SNOMED               98720
SNOMED Veterinary     3093
"""

nonstd_conditions = nonstd_concept[nonstd_concept.domain_id == 'Condition']
nonstd_conditions.groupby('vocabulary_id')['concept_id'].count()
"""
CDISC                   455
CIEL                  38818
CIM10                 13885
CO-CONNECT               16
Cohort                   66
HemOnc                  260
ICD10                 14113
ICD10CM               88510
ICD10CN               30588
ICD10GM               15952
ICD9CM                14929
ICDO3                  5677
KCD7                  19705
MeSH                  12343
Nebraska Lexicon     150062
OMOP Extension            8
OPCS4                     5
OXMIS                  5704
OncoTree                885
PPI                      74
Read                  47836
SMQ                     324
SNOMED                58172
SNOMED Veterinary       144
"""

# nonstd_concept[nonstd_concept['vocabulary_id'] == 'CIEL'].concept_name


# Extract a mapping of standard concepts to non-standard names
nonstd_names = extract_nonstd_names(concept, concept_relationship) 

# Get concept synonyms
synonum_names = extract_synonym(concept, concept_synonym)

# Get UMLS definitions
umls_names = extract_umls_description(concept, mrconso_df, mrdef_df)

# Define essential columns to keep from standard concepts
column_keep = ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id', 'concept_code']

# Merge all concept information
# Performs left joins to preserve all standard concepts while adding additional naming information
conceptML = pd.merge(
    std_concept[column_keep],
    nonstd_names,
    on = 'concept_id',
    how = 'left'
).merge(
    synonum_names,
    on = 'concept_id',
    how = 'left'
).merge(
    umls_names,
    on = 'concept_id',
    how = 'left'
).rename(columns={
    'umls_desc': 'descriptions',
    'concept_synonym_name': 'synonym_name'}
)


## combine all names into one
# columns_combine = ['nonstd_name', 'concept_synonym_name', 'umls_desc']
# conceptML1 = conceptML.copy()  
# conceptML1['text'] = conceptML1[columns_combine].apply(lambda x: [i for k in x if isinstance(k, list) for i in k], axis=1)

# Save the final concept mapping table
conceptML.to_feather('data/ML/conceptML1.feather')


