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
from modules.performance import map_concepts, performance_metrics
from modules.ML_sampling import add_special_token
import json
from itertools import chain




model = 'all-MiniLM-L6-v2'
model_path = f'models/{model}'
model_train = SentenceTransformer(model_path)

####################################
## Work on conditions domain
## CIM vocabulary maps to standard concepts
####################################

# load 
#
# positive_dataset_matching.to_feather('data/ML/matching/positive_dataset_matching.feather')
# candidate_df_matching.to_feather('data/ML/matching/candidate_dataset_matching.feather')


positive_dataset_matching = pd.read_feather('data/ML/matching/positive_dataset_matching.feather')
candidate_df_matching = pd.read_feather('data/ML/matching/candidate_dataset_matching.feather')

# all standard concepts used in positive_dataset_matching
std_concept_matching = positive_dataset_matching[['concept_id1', 'sentence1']].drop_duplicates().rename(
    columns={'concept_id1': 'concept_id', 'sentence1': 'concept_name'})

# build the reference embedding for standard concepts

db = ChromaVecDB(model=model_train, name="ref")
db.empty_collection()
db.store_concepts(std_concept_matching, batch_size= 5461)

## for each item in candidate_df_matching
n_results = 200
results = db.query(
        std_concept_matching,
        n_results = n_results
    )

candidate_fp = std_concept_matching.copy()

candidate_fp['maps_to'] = [[int(i) for i in x] for x in results['ids']]
candidate_fp['distance'] = results['distances']


## for each row, sort maps_to by distance
candidate_fp['sorted_maps_to'] = candidate_fp.apply(
    lambda x: [j for i, j in sorted(zip(x['distance'], x['maps_to']), key=lambda pair: pair[0])],
    axis=1
)

candidate_fp['sorted_distance'] = candidate_fp.apply(
    lambda x: [i for i, j in sorted(zip(x['distance'], x['maps_to']), key=lambda pair: pair[0])],
    axis=1
)

candidate_fp = candidate_fp.drop(
    columns=['maps_to', 'distance']
    ).rename(
    columns={
        'sorted_maps_to': 'maps_to', 
        'sorted_distance': 'distance'}
)

## remove the concept_id from sorted_maps_to
## also remove the corresponding distance
candidate_fp['distance'] = candidate_fp.apply(
    lambda x: [x['distance'][i] for i in range(len(x['maps_to'])) if x['maps_to'][i] != x['concept_id']],
    axis=1
)
candidate_fp['maps_to'] = candidate_fp.apply(
    lambda x: [i for i in x['maps_to'] if i != x['concept_id']],
    axis=1
)


candidate_fp = candidate_fp.explode(['maps_to', 'distance'], ignore_index=False)
candidate_fp['within_group_index'] = candidate_fp.groupby(level=0).cumcount()

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

print(f"candidate_fp.shape: {candidate_fp.shape}")

candidate_fp.to_feather('data/ML/matching/candidate_fp_matching.feather')

candidate_fp.columns