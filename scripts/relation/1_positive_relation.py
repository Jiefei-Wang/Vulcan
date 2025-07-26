import pandas as pd
import os
from modules.TOKENS import TOKENS

omop_base_path = "data/omop_feather"
matching_base_path = "data/matching"
relation_base_path = "data/relation"

if not os.path.exists(relation_base_path):
    os.makedirs(relation_base_path)

target_concepts = pd.read_feather(os.path.join(matching_base_path, 'std_condition_concept.feather'))
concept_ancestors = pd.read_feather(os.path.join(omop_base_path, 'concept_ancestor.feather'))

# We want both ancestor and descendant to be in target_concepts
target_ancestors = concept_ancestors[
    concept_ancestors['ancestor_concept_id'].isin(target_concepts['concept_id'])&
    concept_ancestors['descendant_concept_id'].isin(target_concepts['concept_id'])
    ].reset_index(drop=True)

name_bridge_relation = target_ancestors.rename(
    columns={
        'ancestor_concept_id': 'name_id',
        'descendant_concept_id': 'concept_id'
    }
    )[[ 'concept_id', 'name_id']].reset_index(drop=True)

name_table_relation = target_concepts.rename(
    columns={
        'concept_id': 'name_id',
        'concept_name': 'name'
    }
    )[[ 'name_id', 'name']]

name_table_relation['source'] = "OMOP"
name_table_relation['source_id'] = name_table_relation['name_id']
name_table_relation['type'] = "ancestor"

name_table_relation['name'] = TOKENS.child + name_table_relation['name'] 


name_table_relation = name_table_relation.reset_index(drop=True)


name_table_relation.to_feather(os.path.join(relation_base_path, 'name_table_relation.feather'))
name_bridge_relation.to_feather(os.path.join(relation_base_path, 'name_bridge_relation.feather'))

