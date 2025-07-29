import pandas as pd
import os
from modules.timed_logger import logger
from modules.FalsePositives import getFalsePositives
from modules.ModelFunctions import get_ST_model
from modules.CodeBlockExecutor import trace, tracedf

logger.reset_timer()
logger.log("Building False Positive dataset for matching")

output_dir = "data/matching"


concept= pd.read_feather('data/omop_feather/concept.feather')

target_concepts = pd.read_feather(os.path.join(output_dir, 'target_concepts.feather'))
condition_matching_valid_pos = pd.read_feather(os.path.join(output_dir, 'condition_matching_valid_pos.feather'))
condition_matching_test_pos = pd.read_feather(os.path.join(output_dir, 'condition_matching_test_pos.feather'))
condition_matching_name_bridge_train = pd.read_feather(os.path.join(output_dir, 'condition_matching_name_bridge_train.feather'))
condition_matching_name_table_train = pd.read_feather(os.path.join(output_dir, 'condition_matching_name_table_train.feather'))

condition_matching_name_table_train[condition_matching_name_table_train.name_id==1128]
## sentence1(concept_name), sentence2 (name), concept_id1, concept_id2
condition_matching_valid_pos_pair = condition_matching_valid_pos.rename(
    columns={
        'concept_id1': 'corpus_id',
        'concept_id2': 'query_id',
        'sentence1': 'corpus_name',
        'sentence2': 'query_name'
    }
)

condition_matching_test_pos_pair = condition_matching_test_pos.rename(
    columns={
        'concept_id1': 'corpus_id',
        'concept_id2': 'query_id',
        'sentence1': 'corpus_name',
        'sentence2': 'query_name'
    }
)


model, _ = get_ST_model()



####################
## Construct False Positive dataset for training
####################
n_fp_matching = 50

tracedf(target_concepts)
#> DataFrame dimensions: 160288 rows × 2 columns
#> Column names:
#> ['concept_id', 'concept_name']
#> Estimated memory usage: 15.35 MB

model, tokenizer = get_ST_model()

matching_fp = getFalsePositives(
    model=model,
    corpus_ids = target_concepts['concept_id'],
    corpus_names=target_concepts['concept_name'],
    n_fp=n_fp_matching,
    repos='target_concepts_initial_model'
)
matching_fp = matching_fp.sort_values(by='query_id').reset_index(drop=True)

matching_fp.to_feather(os.path.join(output_dir, f'matching_fp_{n_fp_matching}.feather'))
matching_fp.iloc[0:200].to_excel(os.path.join(output_dir, f'matching_fp_{n_fp_matching}.xlsx'), index=False)

tracedf(matching_fp)
#> DataFrame dimensions: 7892160 rows × 6 columns
#> Column names:
#> ['query_id', 'query_name', 'corpus_id', 'corpus_name', 'score', 'label']
#> Estimated memory usage: 1.62 GB



####################
## Add false positive pairs to a subset of train, the valid and test data for evaluation purpose
####################
n_fp_eval= 5
## get a sample of the training data to use for evaluating loss
query_train = condition_matching_name_table_train[['name_id','name']].sample(n=500, random_state=42)

condition_matching_train_subset_fp = getFalsePositives(
        model = model,
        corpus_ids=target_concepts['concept_id'],
        corpus_names=target_concepts['concept_name'],
        query_ids=query_train['name_id'],
        query_names=query_train['name'],
        blacklist_from=condition_matching_name_bridge_train['concept_id'],
        blacklist_to=condition_matching_name_bridge_train['name_id'],
        n_fp=n_fp_eval,
        repos='target_concepts_initial_model'
    )

condition_matching_train_subset_pos = query_train.merge(
    condition_matching_name_bridge_train,
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
condition_matching_train_subset_pos['label'] = 1  # Positive pairs


# concept_id2 and sentence2 are the non-standard concept_id and name
query_valid = condition_matching_valid_pos_pair.drop_duplicates(subset=['query_id', 'query_name'])
condition_matching_valid_fp = getFalsePositives(
        model = model,
        corpus_ids=target_concepts['concept_id'],
        corpus_names=target_concepts['concept_name'],
        query_ids=query_valid['query_id'],
        query_names=query_valid['query_name'],
        blacklist_from=condition_matching_valid_pos['concept_id1'],
        blacklist_to=condition_matching_valid_pos['concept_id2'],
        n_fp=n_fp_eval,
        repos='target_concepts_initial_model'
    )

query_test = condition_matching_test_pos_pair.drop_duplicates(subset=['query_id', 'query_name'])
condition_matching_test_fp = getFalsePositives(
        model = model,
        corpus_ids=target_concepts['concept_id'],
        corpus_names=target_concepts['concept_name'],
        query_ids=query_test['query_id'],
        query_names=query_test['query_name'],
        blacklist_from=condition_matching_test_pos['concept_id1'],
        blacklist_to=condition_matching_test_pos['concept_id2'],
        n_fp=n_fp_eval,
        repos='target_concepts_initial_model'
    )



tracedf(condition_matching_train_subset_fp)
#> DataFrame dimensions: 2381 rows × 6 columns
#> Column names:
#> ['query_id', 'query_name', 'corpus_id', 'corpus_name', 'score', 'label']
#> Estimated memory usage: 572.57 KB

tracedf(condition_matching_train_subset_pos)
#> DataFrame dimensions: 579 rows × 5 columns
#> Column names:
#> ['query_name', 'corpus_name', 'query_id', 'corpus_id', 'label']
#> Estimated memory usage: 124.60 KB

tracedf(condition_matching_valid_fp)
#> DataFrame dimensions: 2319 rows × 6 columns
#> Column names:
#> ['query_id', 'query_name', 'corpus_id', 'corpus_name', 'score', 'label']
#> Estimated memory usage: 467.77 KB

tracedf(condition_matching_test_fp)
#> DataFrame dimensions: 20877 rows × 6 columns
#> Column names:
#> ['query_id', 'query_name', 'corpus_id', 'corpus_name', 'score', 'label']
#> Estimated memory usage: 4.13 MB

# Combine positive and false positive pairs
columns = ['query_name', 'corpus_name', 'query_id', 'corpus_id', 'label']
condition_matching_train_subset = pd.concat([condition_matching_train_subset_pos, condition_matching_train_subset_fp[columns]], ignore_index=True)
condition_matching_valid = pd.concat([condition_matching_valid_pos_pair, condition_matching_valid_fp[columns]], ignore_index=True)
condition_matching_test = pd.concat([condition_matching_test_pos_pair, condition_matching_test_fp[columns]], ignore_index=True)

condition_matching_train_subset = condition_matching_train_subset.sort_values(by=['query_id', 'label']).reset_index(drop=True)
condition_matching_valid = condition_matching_valid.sort_values(by=['query_id', 'label']).reset_index(drop=True)
condition_matching_test = condition_matching_test.sort_values(by=['query_id', 'label']).reset_index(drop=True)



condition_matching_train_subset.to_feather(os.path.join(output_dir, 'condition_matching_train_subset.feather'))
condition_matching_valid.to_feather(os.path.join(output_dir, 'condition_matching_valid.feather'))
condition_matching_test.to_feather(os.path.join(output_dir, 'condition_matching_test.feather'))

condition_matching_train_subset.to_excel(os.path.join(output_dir, 'condition_matching_train_subset.xlsx'), index=False)
condition_matching_valid.to_excel(os.path.join(output_dir, 'condition_matching_valid.xlsx'), index=False)
condition_matching_test.to_excel(os.path.join(output_dir, 'condition_matching_test.xlsx'), index=False)

tracedf(condition_matching_train_subset)
#> DataFrame dimensions: 2960 rows × 5 columns
#> Column names:
#> ['query_name', 'corpus_name', 'query_id', 'corpus_id', 'label']
#> Estimated memory usage: 678.72 KB

tracedf(condition_matching_valid)
#> DataFrame dimensions: 2815 rows × 5 columns
#> Column names:
#> ['corpus_name', 'query_name', 'corpus_id', 'query_id', 'label']
#> Estimated memory usage: 541.49 KB

tracedf(condition_matching_test)
#> DataFrame dimensions: 25347 rows × 5 columns
#> Column names:
#> ['corpus_name', 'query_name', 'corpus_id', 'query_id', 'label']
#> Estimated memory usage: 4.79 MB






logger.done()


