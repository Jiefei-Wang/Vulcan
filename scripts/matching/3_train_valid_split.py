import pandas as pd
import os
from modules.FalsePositives import getFalsePositives
from modules.timed_logger import logger
from sklearn.model_selection import train_test_split
from modules.CodeBlockExecutor import trace, tracedf
import duckdb
from modules.FaissDB import build_index, is_initialized, search_similar
from modules.ModelFunctions import get_ST_model

logger.reset_timer()
logger.log("Combining all map_tables")

output_dir = "data/matching"
std_bridge = pd.read_feather("data/omop_feather/std_bridge.feather")
concept= pd.read_feather('data/omop_feather/concept.feather')
matching_map_table = pd.read_feather('data/matching/matching_map_table.feather')



####################
## define the concepts we are interested in training
####################
logger.log("Define standard and non-standard concepts for training")
condition_concept = concept[concept['domain_id'] == 'Condition'].reset_index(drop=True)
std_condition_concept = condition_concept[condition_concept['standard_concept'] == 'S'].reset_index(drop=True)
nonstd_condition_concept = condition_concept[condition_concept['standard_concept'] != 'S'].reset_index(drop=True)


std_condition_concept.to_feather(os.path.join(output_dir, 'std_condition_concept.feather'))

trace(std_condition_concept.shape)
#> (160288, 10)


# define the mapping table for condition domain
condition_matching_map_table = matching_map_table[matching_map_table['concept_id'].isin(std_condition_concept['concept_id'])].reset_index(drop=True)

tracedf(condition_matching_map_table)
#> DataFrame dimensions: 749123 rows × 5 columns
#> Column names:
#> ['concept_id', 'source', 'source_id', 'type', 'name']
#> Estimated memory usage: 205.52 MB

####################
## Exclude the reserved concepts from map_table
####################
logger.log("Exclude the reserved concepts from map_table")
reserved_vocab = "CIEL"

reserved_concepts = nonstd_condition_concept[nonstd_condition_concept.vocabulary_id == reserved_vocab]
reserved_ids = reserved_concepts.concept_id.astype(str)

# breaks the link between the reserved concepts and the standard concepts
row_filter_valid = (condition_matching_map_table.source == 'OMOP')&(condition_matching_map_table.source_id.isin(reserved_ids))

condition_matching_map_valid = condition_matching_map_table[row_filter_valid].reset_index(drop=True)
condition_matching_map_train = condition_matching_map_table[~row_filter_valid].reset_index(drop=True)

tracedf(condition_matching_map_train)
#> DataFrame dimensions: 740140 rows × 5 columns
#> Column names:
#> ['concept_id', 'source', 'source_id', 'type', 'name']
#> Estimated memory usage: 203.34 MB

tracedf(condition_matching_map_valid)
#> DataFrame dimensions: 8983 rows × 5 columns
#> Column names:
#> ['concept_id', 'source', 'source_id', 'type', 'name']
#> Estimated memory usage: 2.18 MB



# unique concept id in the table
trace(std_condition_concept['concept_id'].nunique())
#> 160288
trace(condition_matching_map_train['concept_id'].nunique())
#> 104672

trace(condition_matching_map_train.groupby(['source', 'type'])['concept_id'].nunique())
#> source  type   
#> OMOP    nonstd     83253
#>         synonym    98677
#> UMLS    DEF        19553
#>         STR        49962
#> Name: concept_id, dtype: int64


####################
## Buld index for train data
####################
logger.log(f"Buld index for train data")
unique_names = condition_matching_map_train['name'].unique()
name_to_id = pd.Series(range(1, len(unique_names) + 1), index=unique_names)
condition_matching_map_train['name_id'] = condition_matching_map_train['name'].map(name_to_id)

# matching_map_table['name_id'] = range(1, len(matching_map_table) + 1)

condition_matching_name_bridge_train = condition_matching_map_train[['concept_id', 'name_id']]
condition_matching_name_table_train = condition_matching_map_train[['name_id', 'source', 'source_id', 'type', 'name']].drop_duplicates(subset=['name_id']).reset_index(drop=True)


condition_matching_name_bridge_train.to_feather(os.path.join(output_dir, 'condition_matching_name_bridge_train.feather'))
condition_matching_name_table_train.to_feather(os.path.join(output_dir, 'condition_matching_name_table_train.feather'))

tracedf(condition_matching_name_bridge_train)
#> DataFrame dimensions: 740140 rows × 2 columns
#> Column names:
#> ['concept_id', 'name_id']
#> Estimated memory usage: 11.29 MB

tracedf(condition_matching_name_table_train)
#> DataFrame dimensions: 659459 rows × 5 columns
#> Column names:
#> ['name_id', 'source', 'source_id', 'type', 'name']
#> Estimated memory usage: 178.42 MB


####################
## Create valid and test mapping table
## sentence1, sentence2, concept_id1, concept_id2
####################
logger.log("Split valid and test data")

matching_valid_test = condition_matching_map_valid[['name', 'source', 'source_id', 'concept_id']].copy()
condition_matching_valid_test = matching_valid_test.merge(
    std_condition_concept[['concept_id', 'concept_name']],
    on = 'concept_id',
    how = 'inner'
).rename(
    columns={
        'name': 'sentence1',
        'source_id': 'concept_id1',
        'concept_name': 'sentence2',
        'concept_id': 'concept_id2'
    }
)

condition_matching_valid_test['label'] = 1  # Positive pairs
condition_matching_valid_test['concept_id1'] = condition_matching_valid_test['concept_id1'].astype('int64')
condition_matching_valid_test = condition_matching_valid_test[['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']].reset_index(drop=True)


condition_matching_valid_pos, condition_matching_test_pos = train_test_split(
    condition_matching_valid_test,
    test_size=0.9,
    random_state=42,
    shuffle=True)

condition_matching_valid_pos.reset_index(drop=True, inplace=True)
condition_matching_test_pos.reset_index(drop=True, inplace=True)


tracedf(condition_matching_valid_pos)
#> DataFrame dimensions: 898 rows × 5 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']
#> Estimated memory usage: 161.31 KB

tracedf(condition_matching_test_pos)
#> DataFrame dimensions: 8085 rows × 5 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']
#> Estimated memory usage: 1.42 MB

####################
## Add negative pairs to the valid and test data
####################
if not is_initialized():
    model, _ = get_ST_model()
    build_index(model, std_condition_concept[['concept_id', 'concept_name']], repos='target_concepts_initial_model')


def get_negative_pairs(df, std_condition_concept, n_neg=5):
    df = df.copy().drop_duplicates(subset=['concept_id1', "sentence1"])
    fp = getFalsePositives(
        model = model,
        corpus_names=std_condition_concept[['concept_id', 'concept_name']],
        query_names=df.rename(
            columns={
                'concept_id1': 'concept_id',
                'sentence1': 'concept_name'
            }
        ),
        blacklist=df[['concept_id1', 'concept_id2']],
        n_fp=5,
        repos='target_concepts_initial_model'
    )
    
    return fp



## get a sample of the training data to use for evaluating loss
corpus_train = condition_matching_map_train[['name_id', 'name']].drop_duplicates().rename(
    columns={
        'name_id': 'concept_id',
        'name': 'concept_name'
    }
)
big_offset = 10**20
corpus_train['concept_id'] = corpus_train['concept_id'] + big_offset # offset to avoid collision with the real concept ids
query_train_ids = condition_matching_map_train['concept_id'].drop_duplicates().sample(n=len(condition_matching_valid_pos), random_state=42)
query_train = concept[concept['concept_id'].isin(query_train_ids)][['concept_id', 'concept_name']].reset_index(drop=True)
blacklist_train = condition_matching_map_train[['concept_id', 'name_id']].rename(
    columns={
        'concept_id': 'concept_id1',
        'name_id': 'concept_id2'
    }
)

blacklist_train['concept_id2'] = blacklist_train['concept_id2'] + big_offset

condition_matching_train_subset_fp = getFalsePositives(
        model = model,
        corpus_names=corpus_train,
        query_names=query_train,
        blacklist=blacklist_train,
        n_fp=5,
        repos='condition_matching_train_subset'
    )

condition_matching_train_subset_pos = condition_matching_map_train[condition_matching_map_train['concept_id'].isin(query_train_ids)][['concept_id', 'name', 'name_id']].drop_duplicates(
    subset=['concept_id']).merge(
    query_train,
    on='concept_id'
).rename(
    columns={
        'concept_id': 'concept_id1',
        'concept_name': 'sentence1',
        'name_id': 'concept_id2',
        'name': 'sentence2'
    }
).reset_index(drop=True)
condition_matching_train_subset_pos['label'] = 1  # Positive pairs





condition_matching_valid_fp = get_negative_pairs(
    condition_matching_valid_pos,
    std_condition_concept,
    n_neg=5
)


condition_matching_test_fp = get_negative_pairs(
    condition_matching_test_pos,
    std_condition_concept,
    n_neg=5
)


tracedf(condition_matching_train_subset_fp)
#> DataFrame dimensions: 2921 rows × 6 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'score', 'label']
#> Estimated memory usage: 608.59 KB

tracedf(condition_matching_valid_fp)
#> DataFrame dimensions: 4053 rows × 6 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'score', 'label']
#> Estimated memory usage: 791.60 KB

tracedf(condition_matching_test_fp)
#> DataFrame dimensions: 36508 rows × 6 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'score', 'label']
#> Estimated memory usage: 6.96 MB



columns = ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']
condition_matching_train_subset = pd.concat([condition_matching_train_subset_pos, condition_matching_train_subset_fp[columns]], ignore_index=True)
condition_matching_valid = pd.concat([condition_matching_valid_pos, condition_matching_valid_fp[columns]], ignore_index=True)
condition_matching_test = pd.concat([condition_matching_test_pos, condition_matching_test_fp[columns]], ignore_index=True)


condition_matching_train_subset.to_feather(os.path.join(output_dir, 'condition_matching_train_subset.feather'))

condition_matching_valid.to_feather(os.path.join(output_dir, 'condition_matching_valid.feather'))
condition_matching_test.to_feather(os.path.join(output_dir, 'condition_matching_test.feather'))


tracedf(condition_matching_valid)
#> DataFrame dimensions: 4951 rows × 5 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']
#> Estimated memory usage: 921.17 KB

tracedf(condition_matching_test)
#> DataFrame dimensions: 44593 rows × 5 columns
#> Column names:
#> ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']
#> Estimated memory usage: 8.11 MB

logger.done()








