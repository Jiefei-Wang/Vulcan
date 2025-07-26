import pandas as pd
import os
from modules.TOKENS import TOKENS
from modules.ModelFunctions import get_ST_model
from modules.CodeBlockExecutor import trace, tracedf
from modules.FalsePositives import get_false_positives

omop_base_path = "data/omop_feather"
matching_base_path = "data/matching"
relation_base_path = "data/relation"


target_concepts = pd.read_feather(os.path.join(matching_base_path, 'std_condition_concept.feather'))

name_table_relation = pd.read_feather(os.path.join(relation_base_path, 'name_table_relation.feather'))
name_bridge_relation = pd.read_feather(os.path.join(relation_base_path, 'name_bridge_relation.feather'))


corpus = name_table_relation[['name_id', 'name']].copy().rename(columns={'name_id': 'concept_id', 'name': 'concept_name'})
query = target_concepts[['concept_id', 'concept_name']].copy()

blacklist1 = name_bridge_relation[['concept_id', 'name_id']].copy().rename(columns={'concept_id': 'concept_id1', 'name_id': 'concept_id2'})

blacklist2 = target_concepts[['concept_id']].copy().rename(columns={'concept_id': 'concept_id1'})
blacklist2['concept_id2'] = blacklist2['concept_id1']

blacklist = pd.concat([blacklist1, blacklist2], ignore_index=True).drop_duplicates().reset_index(drop=True)


tracedf(corpus)
#> DataFrame dimensions: 160288 rows × 2 columns
#> Column names:
#> ['concept_id', 'concept_name']
#> Estimated memory usage: 17.18 MB

tracedf(query)
#> DataFrame dimensions: 160288 rows × 2 columns
#> Column names:
#> ['concept_id', 'concept_name']
#> Estimated memory usage: 15.35 MB

tracedf(blacklist)
#> DataFrame dimensions: 5418029 rows × 2 columns
#> Column names:
#> ['concept_id1', 'concept_id2']
#> Estimated memory usage: 82.67 MB




base_model = 'ClinicalBERT'
model, tokenizer = get_ST_model(base_model)

n_fp_relation = 50

fp_relation = get_false_positives(
    model=model,
    corpus_concepts=corpus,
    query_concepts=query,
    n_fp=n_fp_relation,
    blacklist=blacklist,
    repos='relation_corpus_model'
)
fp_relation.to_feather(os.path.join(relation_base_path, f'fp_relation_{n_fp_relation}.feather'))


trace(fp_relation.iloc[0])

trace(fp_relation.iloc[0:5])