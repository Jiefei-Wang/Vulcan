## This script convert the OMOP concept from csv to feather format
## Key modifications:
##  1. Add a column 'std_concept_id' to the concept table.
##  2. Remove the concept with no mapping to standard concept.


import pandas as pd

# Read dataset
concept = pd.read_csv('data/omop_csv/CONCEPT.csv', delimiter='\t', low_memory=False, na_filter=False)
concept_relationship = pd.read_csv('data/omop_csv/CONCEPT_RELATIONSHIP.csv', delimiter='\t', low_memory=False, na_filter=False)

std_concepts = concept[concept['standard_concept'] == 'S']
nonstd_concepts = concept[concept['standard_concept'] != 'S']


## merge nonstd_conditions with concept_relationship to get the standard_concept_id
std_map = concept_relationship[concept_relationship.relationship_id=='Maps to'][['concept_id_1', 'concept_id_2']].rename(columns={'concept_id_2': 'std_concept_id'})

## map, for multple mappings, combine them into a list
concept_merged = nonstd_concepts.merge(
    std_map, left_on='concept_id', right_on='concept_id_1', 
    how='left'
    ).drop(columns=['concept_id_1'])


## filter out the concept with no mapping
concept_merged = concept_merged[concept_merged.std_concept_id.apply(lambda x: not pd.isna(x))]

## type to int64
concept_merged.std_concept_id = concept_merged.std_concept_id.astype('int64')

## group by concept_id and combine the standard_concept_id into a list
concept_merged = concept_merged.groupby('concept_id'
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
conceptEX = pd.concat([std_concepts, concept_merged])

## set concept_id as index
conceptEX = conceptEX.set_index('concept_id')

conceptEX.to_feather('data/omop_feather/conceptEX.feather')
concept_relationship.to_feather('data/omop_feather/concept_relationship.feather')