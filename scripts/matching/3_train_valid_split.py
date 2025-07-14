import pandas as pd
import os
from modules.timed_logger import logger
from sklearn.model_selection import train_test_split
from modules.CodeBlockExecutor import trace, tracedf
import duckdb
from modules.FaissDB import build_index, is_initialized, search_similar
from modules.ModelFunctions import load_ST_model

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


# define the mapping table for condition domain
condition_matching_map_table = matching_map_table[matching_map_table['concept_id'].isin(std_condition_concept['concept_id'])].reset_index(drop=True)

tracedf(condition_matching_map_table)

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

tracedf(condition_matching_map_valid)



# unique concept id in the table
trace(std_condition_concept['concept_id'].nunique())
trace(condition_matching_map_train['concept_id'].nunique())

trace(condition_matching_map_train.groupby(['source', 'type'])['concept_id'].nunique())


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

tracedf(condition_matching_name_table_train)


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

tracedf(condition_matching_test_pos)

####################
## Add negative pairs to the valid and test data
####################
if not is_initialized():
    model, _ = load_ST_model()
    build_index(model, std_condition_concept[['concept_id', 'concept_name']])


def get_negative_pairs(df, n_neg=5):
    df = df.copy().drop_duplicates(subset=['concept_id1', "sentence1"])
    
    query_concept_ids = df['concept_id1'].tolist()
    query_texts = df['sentence1'].tolist()
    search_results = search_similar(query_concept_ids, query_texts, top_k=n_neg)

    # Exclude concept_id1, concept_id2 pairs in the df from search_results
    # as they are positive pairs
    search_results = duckdb.query("""
        SELECT query_concept_id as concept_id1, query_text as sentence1, concept_id as concept_id2, concept_name as sentence2, score
        FROM search_results
        ANTI JOIN df
        ON search_results.query_concept_id = df.concept_id1
        AND search_results.concept_id = df.concept_id2
        where score <=0.99
    """).df()
    
    search_results['label'] = 0  # Negative pairs
    return search_results


condition_matching_valid_neg = get_negative_pairs(condition_matching_valid_pos, n_neg=5)
condition_matching_test_neg = get_negative_pairs(condition_matching_test_pos, n_neg=5)

tracedf(condition_matching_valid_neg)

tracedf(condition_matching_test_neg)



columns = ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']
condition_matching_valid = pd.concat([condition_matching_valid_pos, condition_matching_valid_neg[columns]], ignore_index=True)
condition_matching_test = pd.concat([condition_matching_test_pos, condition_matching_test_neg[columns]], ignore_index=True)

condition_matching_valid.to_feather(os.path.join(output_dir, 'condition_matching_valid.feather'))

condition_matching_test.to_feather(os.path.join(output_dir, 'condition_matching_test.feather'))


tracedf(condition_matching_valid)

tracedf(condition_matching_test)

logger.done()








