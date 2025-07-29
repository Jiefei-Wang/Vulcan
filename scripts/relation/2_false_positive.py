import pandas as pd
import os
from modules.TOKENS import TOKENS
from modules.ModelFunctions import get_ST_model, auto_load_model
from modules.CodeBlockExecutor import trace, tracedf
from modules.FalsePositives import getFalsePositives

omop_base_path = "data/omop_feather"
matching_base_path = "data/matching"
relation_base_path = "data/relation"


target_concepts = pd.read_feather(os.path.join(matching_base_path, 'target_concepts.feather'))
name_table_relation = pd.read_feather(os.path.join(relation_base_path, 'name_table_relation.feather'))
name_bridge_relation = pd.read_feather(os.path.join(relation_base_path, 'name_bridge_relation.feather'))


tracedf(target_concepts)
#> DataFrame dimensions: 160288 rows × 2 columns
#> Column names:
#> ['concept_id', 'concept_name']
#> Estimated memory usage: 15.35 MB

# relation_base_model = "output/matching_model"
relation_base_model = "output/relation_with_token"
# model, tokenizer = get_ST_model(relation_base_model)
model, tokenizer, _ = auto_load_model(relation_base_model)
n_fp_relation = 20



fp_relation = getFalsePositives(
    model=model,
    corpus_ids=target_concepts['concept_id'],
    corpus_names=target_concepts['concept_name'],
    query_ids=name_table_relation['name_id'],
    query_names=name_table_relation['name'],
    n_fp=n_fp_relation,
    blacklist_from=name_bridge_relation['concept_id'],
    blacklist_to=name_bridge_relation['name_id'],
    repos='target_concepts_initial_model'
)
# sort by query_id
fp_relation = fp_relation.sort_values(by='query_id').reset_index(drop=True)

fp_relation.to_feather(os.path.join(relation_base_path, f'fp_relation_{n_fp_relation}.feather'))
fp_relation.iloc[0:200].to_excel(os.path.join(relation_base_path, f'fp_relation_{n_fp_relation}.xlsx'), index=False)


tracedf(fp_relation)
#> DataFrame dimensions: 2979773 rows × 6 columns
#> Column names:
#> ['query_id', 'query_name', 'corpus_id', 'corpus_name', 'score', 'label']
#> Estimated memory usage: 624.58 MB

trace(fp_relation.iloc[0])
#> query_id                                                   22274
#> query_name     <|parent of|>Neoplasm of uncertain behavior of...
#> corpus_id                                               37206233
#> corpus_name                             Disorder of joint region
#> score                                                   0.539325
#> label                                                          0
#> Name: 0, dtype: object

trace(fp_relation.iloc[0:5])
#>    query_id                                         query_name  corpus_id                                        corpus_name     score  label
#> 0     22274  <|parent of|>Neoplasm of uncertain behavior of...   37206233                           Disorder of joint region  0.539325      0
#> 1     22274  <|parent of|>Neoplasm of uncertain behavior of...   40485387                        Neoplasm of skeletal system  0.547568      0
#> 2     22274  <|parent of|>Neoplasm of uncertain behavior of...    4260918                          Disorder of spinal region  0.543581      0
#> 3     22274  <|parent of|>Neoplasm of uncertain behavior of...    4055266                             Neoplasm of vocal cord  0.542198      0
#> 4     22274  <|parent of|>Neoplasm of uncertain behavior of...   43021240  Complication associated with musculoskeletal i...  0.555451      0





####################
## Add false positive pairs to a subset of train, the valid and test data for evaluation purpose
####################
n_fp_eval= 5
## get a sample of the training data to use for evaluating loss
query_train = name_table_relation[['name_id','name']].sample(n=500, random_state=42)

condition_relation_train_subset_fp = getFalsePositives(
        model = model,
        corpus_ids=target_concepts['concept_id'],
        corpus_names=target_concepts['concept_name'],
        query_ids=query_train['name_id'],
        query_names=query_train['name'],
        blacklist_from=name_bridge_relation['concept_id'],
        blacklist_to=name_bridge_relation['name_id'],
        n_fp=n_fp_eval,
        repos='target_concepts_initial_model'
    )

condition_relation_train_subset_pos = query_train.merge(
    name_bridge_relation,
    on='name_id'
    ).merge(
        target_concepts,
        on='concept_id'
    ).rename(
        columns={
            'name': 'query_name',
            'concept_name': 'corpus_name',
            'name_id': 'query_id',
            'concept_id': 'corpus_id'
        }
    )[['query_name', 'corpus_name', 'query_id', 'corpus_id']].reset_index(drop=True)
condition_relation_train_subset_pos['label'] = 1  # Positive pairs

tracedf(condition_relation_train_subset_fp)
#> DataFrame dimensions: 2110 rows × 6 columns
#> Column names:
#> ['query_id', 'query_name', 'corpus_id', 'corpus_name', 'score', 'label']
#> Estimated memory usage: 454.02 KB

tracedf(condition_relation_train_subset_pos)
#> DataFrame dimensions: 3431 rows × 5 columns
#> Column names:
#> ['query_name', 'corpus_name', 'query_id', 'corpus_id', 'label']
#> Estimated memory usage: 707.65 KB



condition_relation_train_subset = pd.concat(
    [condition_relation_train_subset_pos, condition_relation_train_subset_fp],
    ignore_index=True
).reset_index(drop=True).drop(columns=['score'])

condition_relation_train_subset = condition_relation_train_subset.sort_values(by=['query_id', 'label']).reset_index(drop=True)


tracedf(condition_relation_train_subset)
#> DataFrame dimensions: 5541 rows × 5 columns
#> Column names:
#> ['query_name', 'corpus_name', 'query_id', 'corpus_id', 'label']
#> Estimated memory usage: 1.12 MB

condition_relation_train_subset.to_feather(os.path.join(relation_base_path, 'condition_relation_train_subset.feather'))

condition_relation_train_subset.to_excel(os.path.join(relation_base_path, 'condition_relation_train_subset.xlsx'), index=False)
