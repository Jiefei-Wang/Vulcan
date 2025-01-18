# This file is used to generate the mapping from 
# standard to non-standard concepts
#
# For each standard concept, create columns:
# - nonstd_ids: list of non-standard concept ids
# - nonstd_names: list of non-standard concept names (size = nonstd_id)
# - descriptions: list of non-standard concept descriptions (from UMLS) (size = nonstd_id)
# - synonym_names: list of concept synonyms (if no synonym, then empty list)
# 
# Store the result in data/ML/conceptML.feather
import importlib
from tqdm import tqdm
import pandas as pd
from modules.ML_extract_name import extract_nonstd_names, extract_synonym, extract_umls_description
from modules.UMLS_file import read_mrconso, read_mrdef

importlib.reload(modules.ML_extract_name)


## Read OMOP datasets
concept = pd.read_feather('data/omop_feather/concept.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')
concept_synonym = pd.read_feather("data/omop_feather/concept_synonym.feather")


## Read UMLS datasets
## UMLS data contains descriptions of the concepts
mrconso_path = "data/UMLS_raw/MRCONSO.RRF"
umls_def_path = "data/UMLS_raw/MRDEF.RRF"
mrconso_df = read_mrconso(mrconso_path)
mrdef_df = read_mrdef(umls_def_path)


## Obtain concept information
std_concept = concept[concept['standard_concept'] == 'S']
nonstd_names = extract_nonstd_names(concept, concept_relationship)

synonum_names = extract_synonym(concept, concept_synonym)
umls_names = extract_umls_description(concept, mrconso_df, mrdef_df)


column_keep = ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id', 'concept_code']
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
)

print(conceptML.columns)
# print(conceptML[conceptML['concept_id']==8715].values)

## combine all names into one
# columns_combine = ['nonstd_name', 'concept_synonym_name', 'umls_desc']
# conceptML1 = conceptML.copy()  
# conceptML1['text'] = conceptML1[columns_combine].apply(lambda x: [i for k in x if isinstance(k, list) for i in k], axis=1)

conceptML.to_feather('data/ML/conceptML1.feather')


