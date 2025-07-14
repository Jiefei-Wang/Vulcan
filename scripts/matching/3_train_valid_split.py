import pandas as pd
import os
from modules.timed_logger import logger
from sklearn.model_selection import train_test_split
import duckdb

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


# define the mapping table for condition domain
condition_matching_map_table = matching_map_table[matching_map_table['concept_id'].isin(std_condition_concept['concept_id'])].reset_index(drop=True)


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





# unique concept id in the table
std_condition_concept['concept_id'].nunique() # 160288
condition_matching_map_train['concept_id'].nunique()  # 104672

condition_matching_map_train.groupby(['source', 'type'])['concept_id'].nunique()
# source  type   
# OMOP    nonstd     83257
#         synonym    98677
# UMLS    DEF        19553
#         STR        49962


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
# [1624671 rows x 2 columns]
condition_matching_name_table_train.to_feather(os.path.join(output_dir, 'condition_matching_name_table_train.feather'))
# [764392 rows x 5 columns]



####################
## Create valid and test mapping table
## sentence1, sentence2, concept_id1, concept_id2
####################
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

condition_matching_valid_test = condition_matching_valid_test[['sentence1', 'sentence2', 'concept_id1', 'concept_id2']].reset_index(drop=True)


## split the valid and test data
logger.log("Split valid and test data")

condition_matching_valid, condition_matching_test = train_test_split(
    condition_matching_valid_test,
    test_size=0.9,
    random_state=42,
    shuffle=True)

condition_matching_valid.reset_index(drop=True, inplace=True)
condition_matching_test.reset_index(drop=True, inplace=True)


condition_matching_valid.to_feather(os.path.join(base_path, 'condition_matching_valid.feather'))

condition_matching_test.to_feather(os.path.join(base_path, 'condition_matching_test.feather'))
# [30737 rows x 4 columns]
logger.done()





from sentence_transformers import SentenceTransformer


query_texts = condition_matching_valid['sentence1'].tolist()
corpus_texts = std_condition_concept['concept_name'].tolist()[1:100]
model = SentenceTransformer('models/ClinicalBERT')  # Fast and good quality
query_embeddings = model.encode(query_texts, normalize_embeddings=True)
corpus_embeddings = model.encode(corpus_texts, normalize_embeddings=True)


import faiss

# Build index
dimension = query_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # inner product = cosine if normalized
index.add(corpus_embeddings.astype('float32'))

# Search
top_k = 5
scores, indices = index.search(query_embeddings.astype('float32'), top_k)
