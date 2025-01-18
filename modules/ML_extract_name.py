import pandas as pd

def extract_nonstd_names(concept, concept_relationship):
    """
    Extract non-standard names for each standard concept
    
    Args:
    concept: pd.DataFrame, concept table
    concept_relationship: pd.DataFrame, concept_relationship table
    
    Returns:
    pd.DataFrame: concept_id, nonstd
    """
    print("Running new version of function")
    std_concepts = concept[concept['standard_concept'] == 'S']
    
    ## for each standard concept, get all non-standard codes
    nonstd_map = concept_relationship[concept_relationship.relationship_id=='Maps to'][['concept_id_1', 'concept_id_2']].rename(columns={'concept_id_1': 'nonstd_concept_id'})
    nonstd_map = nonstd_map[nonstd_map.nonstd_concept_id!=nonstd_map.concept_id_2]

    concept_merged = std_concepts.merge(
        nonstd_map, left_on='concept_id', right_on='concept_id_2', 
        how='inner'
        ).drop(columns=['concept_id_2'])
        
    ## find the non-standard concept names
    nonstd_concept_name_map = concept[['concept_id', 'concept_name']].copy().rename(columns={'concept_name': 'nonstd_name', 'concept_id': 'nonstd_concept_id'})

    concept_merged2 = concept_merged.merge(
        nonstd_concept_name_map, 
        on='nonstd_concept_id', 
        how='inner')

    ## aggregate, each concept can have multiple nonstandard concept name
    concept_merged3 = concept_merged2.groupby('concept_id'
        ).agg({
            'nonstd_name': lambda x: list(x),
            'nonstd_concept_id': lambda x: list(x)
        }).reset_index()
        
    print(concept_merged3.columns)
        
    return concept_merged3


def extract_synonym(concept, concept_synonym):
    """
    Extract synonyms for each standard
    
    Args:
    concept: pd.DataFrame, concept table
    concept_synonym: pd.DataFrame, concept_synonym table
    
    Returns:
    pd.DataFrame: concept_id, concept_synonym_name
    """
    std_concepts = concept[concept['standard_concept'] == 'S']
    
    concept_other_names = concept_synonym.merge(
        std_concepts[['concept_id']],
        on='concept_id',
        how='inner'
    )
    
    concept_other_names = concept_other_names.groupby('concept_id'
        ).agg({
            'concept_synonym_name': lambda x: list(x)}
        ).reset_index()
        
    return concept_other_names
    

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

def extract_umls_description(std_concepts, mrconso_df, mrdef_df):
    """
    Extract UMLS descriptions for each standard concept
    
    Args:
    std_concepts: pd.DataFrame, standard concepts
    mrconso_df: pd.DataFrame, MRCONSO.RRF
    mrdef_df: pd.DataFrame, MRDEF.RRF
    
    Returns:
    pd.DataFrame: concept_id, umls_desc
    """
    std_concepts = std_concepts[std_concepts['standard_concept'] == 'S'].copy()

    std_concepts['SAB'] = std_concepts['vocabulary_id'].map(OMOP_TO_UMLS)

    ## remove NAN from SAB
    concept_mapped = std_concepts.dropna(subset=['SAB']).copy()
    concept_mapped['concept_code'] = concept_mapped['concept_code'].astype(str)
    mrconso_df['CODE'] = mrconso_df['CODE'].astype(str)

    # Merge conceptEX with mrconso_df to get the CUI
    merged_df = pd.merge(
        concept_mapped,
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
    def_list_df2['DEF'] = def_list_df2['DEF'].apply(lambda x: list(set(x)))
    
    ## remove 0 length definitions
    def_list_df3 = def_list_df2[def_list_df2['DEF'].apply(len) > 0].copy()
    def_list_df3.rename(columns={'DEF': 'umls_desc'}, inplace=True)
    return def_list_df3
