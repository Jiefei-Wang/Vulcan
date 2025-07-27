import pandas as pd
import os
from modules.TOKENS import TOKENS
from modules.CodeBlockExecutor import trace, tracedf

omop_base_path = "data/omop_feather"
matching_base_path = "data/matching"
relation_base_path = "data/relation"

if not os.path.exists(relation_base_path):
    os.makedirs(relation_base_path)

target_concepts = pd.read_feather(os.path.join(matching_base_path, 'target_concepts.feather'))
concept_ancestors = pd.read_feather(os.path.join(omop_base_path, 'concept_ancestor.feather'))

# We want both ancestor and descendant to be in target_concepts, they should be directly related
# i.e., min_levels_of_separation == 1
relation_tables = concept_ancestors[
    concept_ancestors['ancestor_concept_id'].isin(target_concepts['concept_id'])&
    concept_ancestors['descendant_concept_id'].isin(target_concepts['concept_id']) &
    (concept_ancestors['min_levels_of_separation'] == 1)
    ].reset_index(drop=True)


name_bridge_relation = relation_tables.rename(
    columns={
        'ancestor_concept_id': 'concept_id',
        'descendant_concept_id': 'name_id'
    }
    )[[ 'concept_id', 'name_id']].reset_index(drop=True)

name_table_relation = target_concepts.rename(
    columns={
        'concept_id': 'name_id',
        'concept_name': 'name'
    }
    )[[ 'name_id', 'name']]

# filter out the names that are not in the name_bridge_relation
name_table_relation = name_table_relation[
    name_table_relation.name_id.isin(name_bridge_relation['name_id'])].reset_index(drop=True)

name_table_relation['source'] = "OMOP"
name_table_relation['source_id'] = name_table_relation['name_id']
name_table_relation['type'] = "child"

name_table_relation['name'] = TOKENS.parent + name_table_relation['name'] 


name_table_relation = name_table_relation.reset_index(drop=True)


name_table_relation.to_feather(os.path.join(relation_base_path, 'name_table_relation.feather'))
name_bridge_relation.to_feather(os.path.join(relation_base_path, 'name_bridge_relation.feather'))

tracedf(name_table_relation)
#> DataFrame dimensions: 159468 rows × 5 columns
#> Column names:
#> ['name_id', 'name', 'source', 'source_id', 'type']
#> Estimated memory usage: 34.75 MB

tracedf(name_bridge_relation)
#> DataFrame dimensions: 370632 rows × 2 columns
#> Column names:
#> ['concept_id', 'name_id']
#> Estimated memory usage: 5.66 MB


# find name_id with most concept_id
name_id_counts = name_bridge_relation['name_id'].value_counts().reset_index()
name_id_counts.columns = ['name_id', 'concept_count']

trace(relation_tables[relation_tables.descendant_concept_id==name_id_counts.name_id[0]])
#>         ancestor_concept_id  descendant_concept_id  min_levels_of_separation  max_levels_of_separation
#> 186304               321876               37204863                         1                         1
#> 186881              4028244               37204863                         1                         1
#> 187014              4027461               37204863                         1                         1
#> 191536               434337               37204863                         1                         1
#> 194491              4301371               37204863                         1                         1
#> 198105              4006473               37204863                         1                         1
#> 200883              4134595               37204863                         1                         1
#> 202685             40484120               37204863                         1                         1
#> 206157              4180159               37204863                         1                         1
#> 208576             45757752               37204863                         1                         1
#> 209755              1244944               37204863                         1                         1
#> 209991               199870               37204863                         1                         1
#> 211269              4059452               37204863                         1                         1
#> 211922              4180314               37204863                         1                         1
#> 216981              4181326               37204863                         1                         1
#> 217295              4188970               37204863                         1                         1
#> 220864              4188971               37204863                         1                         1
#> 226823              4179992               37204863                         1                         1




