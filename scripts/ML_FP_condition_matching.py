# This script generate false postives samples for training data


## For Windows: Python 3.10, chromaDB version 0.5.4
## For Windows: Python 3.11, chromadb==0.5.0 chroma-hnswlib==0.7.3
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

## without this, conda might give an error when loading chromadb
import onnxruntime

import pandas as pd
from sentence_transformers import SentenceTransformer
from modules.ChromaVecDB import ChromaVecDB

from modules.timed_logger import logger
logger.reset_timer()


model = 'all-MiniLM-L6-v2'
model_path = f'models/{model}'
model_train = SentenceTransformer(model_path)

####################################
## Work on conditions domain
####################################
logger.log("Loading matching tables")
# std_target = pd.read_feather('data/ML/base_data/std_target.feather')
positive_df_matching = pd.read_feather('data/ML/matching/positive_df_matching.feather')
candidate_df_matching = pd.read_feather('data/ML/matching/candidate_df_matching.feather')

# all standard concepts used in positive_df_matching
std_concept_matching = positive_df_matching[['concept_id1', 'sentence1']].drop_duplicates().rename(
    columns={'concept_id1': 'concept_id', 'sentence1': 'concept_name'})

# build the reference embedding for standard concepts


logger.log("Building reference embedding for standard concepts")
db = ChromaVecDB(model=model_train, name="ref")
db.empty_collection()
db.store_concepts(std_concept_matching, batch_size= 5461)

logger.log("Query the reference embedding for standard concepts")
## for each item in candidate_df_matching
n_results = 200
results = db.query(
        std_concept_matching,
        n_results = n_results
    )


logger.log("Building candidate false positive dataframe")
candidate_fp = std_concept_matching.copy()
candidate_fp['maps_to'] = [[int(i) for i in x] for x in results['ids']]
candidate_fp['distance'] = results['distances']


def sort_id_by_distance(row):
    pairs = list(zip(row['distance'], row['maps_to']))
    if not pairs:
        return ([], []) 
    sorted_pairs = sorted(pairs, key=lambda pair: pair[0])
    sorted_distances, sorted_maps = zip(*sorted_pairs)
    return pd.Series([list(sorted_distances), list(sorted_maps)])


candidate_fp[['distance', 'maps_to']] = candidate_fp.apply(sort_id_by_distance, axis=1)
candidate_fp.iloc[0]
"""
concept_id                                               40664135
concept_name            Condition: Infection present (Deprecated)
maps_to         [40664135, 4269943, 4155635, 4018050, 4017686,...
distance        [2.384185791015625e-07, 0.13260793685913086, 0...
Name: 0, dtype: object
"""

def remove_self_concept(row):
    if row['concept_id'] in row['maps_to']:
        index = row['maps_to'].index(row['concept_id'])
        row['maps_to'].pop(index)
        row['distance'].pop(index)
    return row

candidate_fp = candidate_fp.apply(remove_self_concept, axis=1)

candidate_fp.iloc[0]
"""
concept_id                                               40664135
concept_name            Condition: Infection present (Deprecated)
maps_to         [4269943, 4155635, 4018050, 4017686, 44790687,...
distance        [0.13260793685913086, 0.1543610692024231, 0.15...
Name: 0, dtype: object
"""


candidate_fp = candidate_fp.explode(['maps_to', 'distance'], ignore_index=False)
candidate_fp['within_group_index'] = candidate_fp.groupby(level=0).cumcount()
candidate_fp.columns
# ['concept_id', 'concept_name', 'maps_to', 'distance',
#        'within_group_index']


## link the maps to concept id to its name
candidate_fp = candidate_fp.merge(
    std_concept_matching,
    left_on='maps_to',
    right_on='concept_id',
    how='inner'
).rename(columns={
        'concept_name_x': 'sentence1',
        'concept_name_y': 'sentence2',
        'concept_id_x': 'concept_id1',
        'concept_id_y': 'concept_id2'
    }
).drop(columns=['maps_to'])


candidate_fp.to_feather('data/ML/matching/candidate_fp_matching.feather')

print(f"candidate_fp len: {len(candidate_fp)}") # 20277388
candidate_fp.columns
# ['concept_id1', 'sentence1', 'distance', 'within_group_index',
#        'concept_id2', 'sentence2']

logger.done()