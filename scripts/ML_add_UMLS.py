import pandas as pd
from modules.UMLS_file import read_mrconso, read_mrdef

OMOP_TO_UMLS = {
    'CPT4': 'CPT',
    'ATC': 'ATC',
    'CVX': 'CVX',
    'HCPCS': 'HCPCS',
    'ICD10CM': 'ICD10CM',
    'ICD10PCS': 'ICD10PCS',
    'ICD9CM': 'ICD9CM',
    'ICD9Proc': 'ICD9CM',
    'LOINC': 'LNC',
    'MeSH': 'MSH',
    'NAACCR': 'NAACCR',
    'NDC': 'NDC',
    'NUCC': 'NUCCHCPT',
    'RxNorm': 'RXNORM',
    'SNOMED': 'SNOMEDCT_US',
    'SNOMED Veterinary': 'SNOMEDCT_VET',
    'VANDF': 'VANDF',
    'NDFRT': 'NDFRT',
    'ICD10': 'ICD10',
    'OPS': 'OPS',
    'ICD10GM': 'ICD10GM',
    'JAX': 'HGNC',
    'EDI': 'EDI',
    'MMI': 'MMX',
    'SPL': 'MTHSPL'
}
ULMS_TO_OMOP = {v: k for k, v in OMOP_TO_UMLS.items()}




mrconso_path = "data/UMLS_raw/MRCONSO.RRF"
umls_def_path = "data/UMLS_raw/MRDEF.RRF"
mrconso_df = read_mrconso(mrconso_path)
mrdef_df = read_mrdef(umls_def_path)


conceptEX = pd.read_feather('data/omop_feather/conceptEX.feather')
conceptEX['SAB'] = conceptEX['vocabulary_id'].map(OMOP_TO_UMLS)

## remove NAN from SAB
conceptEX_mapped = conceptEX.dropna(subset=['SAB']).copy()
conceptEX_mapped['concept_code'] = conceptEX_mapped['concept_code'].astype(str)
mrconso_df['CODE'] = mrconso_df['CODE'].astype(str)

# Merge conceptEX with mrconso_df to get the CUI
merged_df = pd.merge(
    conceptEX_mapped,
    mrconso_df[['CUI', 'SAB', 'CODE']],
    left_on=['SAB', 'concept_code'],
    right_on=['SAB', 'CODE'],
    how='inner'
)

# Merge the merged_df with UMLS_def on 'CUI' to get definitions
merged_def_df = pd.merge(
    merged_df,
    mrdef_df[['CUI', 'DEF']],
    on='CUI',
    how='inner'
)

def_list_df = merged_def_df.groupby('concept_id')['DEF'].apply(lambda x: x.dropna().tolist())

## Series to DataFrame
def_list_df2 = def_list_df.reset_index()


## remove 0 length definitions
def_list_df3 = def_list_df2[def_list_df2['DEF'].apply(len) > 0]

# Merge the definitions back into the original conceptEX DataFrame
concept_UMLS = pd.merge(
    conceptEX,
    def_list_df2,
    left_on='concept_id',
    right_on='concept_id',
    how='left'
)


# Rename 'DEF' column to 'explanation'
concept_UMLS.rename(columns={'DEF': 'explanation'}, inplace=True)

# Replace NaN explanations with empty lists
concept_UMLS['explanation'] = concept_UMLS['explanation'].apply(lambda x: x if isinstance(x, list) else [])


concept_UMLS = concept_UMLS.dropna(subset=['SAB']).copy()
concept_UMLS.reset_index(drop=True, inplace=True)

concept_UMLS.to_feather('data/ML/concept_UMLS.feather')