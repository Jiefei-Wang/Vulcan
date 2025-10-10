import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from modules.ModelFunctions import auto_load_model, encode_concepts
from modules.timed_logger import logger
from modules.Dataset import PositiveDataset, NegativeDataset, FalsePositiveDataset, CombinedDataset
from modules.metrics import evaluate_performance
from modules.FaissDB import build_index, is_initialized, delete_repository, search_similar


# exec(open('report/jsm_functions.py').read())

##########################################
# models
# model1: trained with matching only
# model2: trained with matching + relation
##########################################

model1, tokenizer1, train_config1 = auto_load_model("output/finetune_initial/2025-10-02_15-38-11/checkpoint-78690")

# model2, tokenizer2, train_config2 = auto_load_model("output/finetune/2025-07-30_13-33-46")



##########################################
# Load data
##########################################
matching_base_path = "data/matching"
relation_base_path = "data/relation"
omop_base_path = "data/omop_feather"

concept= pd.read_feather('data/omop_feather/concept.feather')
concept_ancestor = pd.read_feather(os.path.join(omop_base_path, 'concept_ancestor.feather'))
std_bridge = pd.read_feather(os.path.join(omop_base_path, 'std_bridge.feather'))

condition_concept = concept[concept['domain_id'] == 'Condition'].reset_index(drop=True)
std_condition_concept = condition_concept[condition_concept['standard_concept'] == 'S'].reset_index(drop=True)
nonstd_condition_concept = condition_concept[condition_concept['standard_concept'] != 'S'].reset_index(drop=True)

target_concepts = pd.read_feather(os.path.join(matching_base_path, 'target_concepts.feather'))

reserved_vocab = "CIEL"
reserved_concepts = nonstd_condition_concept[nonstd_condition_concept.vocabulary_id == reserved_vocab]


std_bridge[std_bridge.concept_id.isin(reserved_concepts['concept_id'])].shape
reserved_concepts = reserved_concepts[reserved_concepts.concept_id.isin(std_bridge.concept_id)]
reserved_concepts = reserved_concepts.reset_index(drop=True)

##########################################
# Generate embeddings for target concepts
##########################################
# if False:
target_concepts['model1_embedding'] = encode_concepts(
    model1, 
    target_concepts.concept_name).tolist()

    # target_concepts['model2_embedding'] = encode_concepts(
    #     model2, 
    #     target_concepts.concept_name).tolist()

#     target_concepts.to_feather('output/tmp/target_concepts_with_embeddings.feather')

# target_concepts = pd.read_feather('output/tmp/target_concepts_with_embeddings.feather')



#########################
# Test on the test dataset
#########################
from modules.FaissDB import build_index, is_initialized, delete_repository, search_similar

query_concepts = reserved_concepts[['concept_id', 'concept_name']]
query_positive_mapping = std_bridge[std_bridge.concept_id.isin(reserved_concepts.concept_id)].reset_index(drop=True).rename(
        columns={
            'concept_id': 'query_id',
            'std_concept_id': 'corpus_id'
        }
    )
query_positive_mapping['label'] = 1

top_k = 100
model1_emb = build_index(
    model = model1, 
    corpus_ids = target_concepts.concept_id, 
    corpus_names = target_concepts.concept_name, 
    corpus_embeddings = target_concepts['model1_embedding'],
    repos='model1'
)

model1_top = search_similar(
    query_ids = query_concepts.concept_id, 
    query_names = query_concepts.concept_name, 
    top_k=top_k, 
    repos='model1')

model1_top = model1_top.merge(
    query_positive_mapping,
    on=['query_id', 'corpus_id'],
    how='left'
) 
model1_top['label'] = model1_top['label'].fillna(0).astype(int)


model1_eval = evaluate_performance(
    query_ids=model1_top['query_id'],
    similarities=model1_top['score'],
    labels=model1_top['label']
)
model1_eval


model2_emb = build_index(
    model = model2,
    corpus_ids = target_concepts.concept_id,
    corpus_names = target_concepts.concept_name,
    corpus_embeddings = target_concepts['model2_embedding'],
    repos='model2'
)


model2_top = search_similar(
    query_ids = query_concepts.concept_id2,
    query_names = query_concepts.sentence2,
    top_k=top_k,
    repos='model2'
)

from modules.TOKENS import TOKENS
parent_query_name = TOKENS.parent + query_concepts.sentence2
model2_parent_top = search_similar(
    query_ids = query_concepts.concept_id2,
    query_names = parent_query_name,
    top_k=top_k,
    repos='model2'
)


# For query_id and corpus_id, if they match concept_id2 and concept_id1 respectively, then label 1, otherwise 0
model1_top = model1_top.merge(
    test_concept_pairs[['concept_id2', 'concept_id1', 'label']].rename(
        columns={
            'concept_id2': 'query_id',
            'concept_id1': 'corpus_id'
        }
    ),
    on=['query_id', 'corpus_id'],
    how='left'
) 
model1_top['label'] = model1_top['label'].fillna(0).astype(int)

model2_top = model2_top.merge(
    test_concept_pairs[['concept_id2', 'concept_id1', 'label']].rename(
        columns={
            'concept_id2': 'query_id',
            'concept_id1': 'corpus_id'
        }
    ),
    on=['query_id', 'corpus_id'],
    how='left'
)
model2_top['label'] = model2_top['label'].fillna(0).astype(int)

relation_tables = concept_ancestors[
    concept_ancestors['ancestor_concept_id'].isin(target_concepts['concept_id'])&
    concept_ancestors['descendant_concept_id'].isin(target_concepts['concept_id'])&
    (concept_ancestors['min_levels_of_separation'] >= 1) &
    (concept_ancestors['min_levels_of_separation'] <= 2)
    ].drop_duplicates(subset=['ancestor_concept_id', 'descendant_concept_id'])[[ 'ancestor_concept_id', 'descendant_concept_id']].reset_index(drop=True)

test_parent_pairs = test_concept_pairs[['concept_id2', 'concept_id1', 'label']].merge(
    relation_tables,
    left_on='concept_id1',
    right_on='descendant_concept_id',
    how='inner'
)[['concept_id2', 'ancestor_concept_id', 'label']].rename(
    columns={
        'ancestor_concept_id': 'concept_id1'
    })


model2_parent_top = model2_parent_top.merge(
    test_parent_pairs[['concept_id2', 'concept_id1', 'label']].rename(
        columns={
            'concept_id2': 'query_id',
            'concept_id1': 'corpus_id'
        }
    ),
    on=['query_id', 'corpus_id'],
    how='left'
)
model2_parent_top['label'] = model2_parent_top['label'].fillna(0).astype(int)
model2_parent_top[['query_id', 'corpus_id', 'label']]
## label category count


model2_eval = evaluate_performance(
    query_ids=model2_top['query_id'],
    similarities=model2_top['score'],
    labels=model2_top['label']
)

model2_parent_eval = evaluate_performance(
    query_ids=model2_parent_top['query_id'],
    similarities=model2_parent_top['score'],
    labels=model2_parent_top['label']
)

from pprint import pprint
pprint(model1_eval)
pprint(model2_eval)
pprint(model2_parent_eval)



model2_top[model2_top.query_id==45946235].iloc[1]

model2_top.to_excel('report/JSM/model2_top.xlsx', index=False)


test_concept_pairs[test_concept_pairs.concept_id2==45907930].iloc[0]


################
# USAGI
################

usagi_top = pd.read_csv('report/JSM/usagi_candidates_export.csv')

usagi_top.columns
# ['source_code', 'source_name', 'target_concept_id', 'target_domain_id', 'match_score']

usagi_top = usagi_top.rename(
    columns={
        'source_code': 'query_id',
        'source_name': 'query_name',
        'target_concept_id': 'corpus_id',
        'match_score': 'score'
    }
).merge(
    test_concept_pairs[['concept_id2', 'concept_id1', 'label']].rename(
        columns={
            'concept_id2': 'query_id',
            'concept_id1': 'corpus_id'
        }
    ),
    on=['query_id', 'corpus_id'],
    how='left'
)

usagi_top['label'] = usagi_top['label'].fillna(0).astype(int)



usagi_eval = evaluate_performance(
    query_ids=usagi_top['query_id'],
    similarities=usagi_top['score'],
    labels=usagi_top['label']
)


from pprint import pprint
pprint(model1_eval)
pprint(model2_eval)
pprint(usagi_eval)