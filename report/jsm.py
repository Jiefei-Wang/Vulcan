import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import math
import numpy as np
import pandas as pd
import wandb
import tempfile
import torch
from tqdm import tqdm

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from modules.ModelFunctions import save_init_model, save_best_model, get_loss, get_base_model, get_ST_model, auto_load_model, encode_concepts
from modules.timed_logger import logger
from modules.Dataset import PositiveDataset, NegativeDataset, FalsePositiveDataset, CombinedDataset
from modules.metrics import evaluate_performance
from modules.FaissDB import build_index, is_initialized, delete_repository, search_similar


exec(open('report/jsm_functions.py').read())

##########################################
# models
# model1: trained with matching only
# model2: trained with matching + relation
##########################################
model1, tokenizer1, train_config1 = auto_load_model("output/matching_model")

model2, tokenizer2, train_config2 = auto_load_model("output/finetune/2025-07-30_13-33-46")



##########################################
# Load data
##########################################
matching_base_path = "data/matching"
relation_base_path = "data/relation"
omop_base_path = "data/omop_feather"

concept_ancestor = pd.read_feather(os.path.join(omop_base_path, 'concept_ancestor.feather'))

# target_concepts = pd.read_feather(os.path.join(matching_base_path, 'target_concepts.feather'))
matching_name_bridge = pd.read_feather(os.path.join(matching_base_path, 'condition_matching_name_bridge_train.feather'))
matching_name_table = pd.read_feather(os.path.join(matching_base_path, 'condition_matching_name_table_train.feather'))
matching_test = pd.read_feather(os.path.join(matching_base_path, 'condition_matching_test.feather'))

name_table_relation = pd.read_feather(os.path.join(relation_base_path, 'name_table_relation.feather'))
name_bridge_relation = pd.read_feather(os.path.join(relation_base_path, 'name_bridge_relation.feather'))
    


##########################################
# Generate embeddings for target concepts
##########################################
if False:
    target_concepts['model1_embedding'] = encode_concepts(
        model1, 
        target_concepts.concept_name).tolist()

    target_concepts['model2_embedding'] = encode_concepts(
        model2, 
        target_concepts.concept_name).tolist()

    target_concepts.to_feather('report/JSM/target_concepts_with_embeddings.feather')

target_concepts = pd.read_feather('report/JSM/target_concepts_with_embeddings.feather')
##########################################
# How many concepts has been used in training?
##########################################
matching_trainable_ids = matching_name_bridge.concept_id.unique()
len(matching_trainable_ids)
# 104198

relation_trainable_ids1 = name_bridge_relation.concept_id.unique()
relation_trainable_ids2 = name_table_relation.name_id.unique()
relation_trainable_ids = np.union1d(relation_trainable_ids1, relation_trainable_ids2)
len(relation_trainable_ids)
# 159777

all_trainable_ids = np.union1d(matching_trainable_ids, relation_trainable_ids)
len(all_trainable_ids)
# 160061

relation_additional_ids = set(relation_trainable_ids) - set(matching_trainable_ids)
len(relation_additional_ids)
# 55863



##########################################
# For each parent concept, find the number of untrained child concepts
# that is trained with relation 
##########################################
parent_child = concept_ancestor[
    concept_ancestor.descendant_concept_id.isin(all_trainable_ids)&
    (concept_ancestor.min_levels_of_separation == 1)
    ][['ancestor_concept_id', 'descendant_concept_id']].reset_index(drop=False).merge(
        target_concepts[['concept_id', 'model1_embedding', 'model2_embedding']].rename(
            columns={
                'concept_id': 'descendant_concept_id',
                'model1_embedding': 'descendant_model1_embedding',
                'model2_embedding': 'descendant_model2_embedding'
            }
            ),
        on='descendant_concept_id',
        how='inner'
    ).merge(
        target_concepts[['concept_id', 'model1_embedding', 'model2_embedding']].rename(
            columns={
                'concept_id': 'ancestor_concept_id',
                'model1_embedding': 'ancestor_model1_embedding',
                'model2_embedding': 'ancestor_model2_embedding'
            }
        ),
        on='ancestor_concept_id',
        how='inner'
    )

parent_child['similarity_model1'] = column_similarity(parent_child['ancestor_model1_embedding'], parent_child['descendant_model1_embedding'])


parent_child['similarity_model2'] = column_similarity(parent_child['ancestor_model2_embedding'], parent_child['descendant_model2_embedding'])


parent_child['label'] = 0
parent_child.loc[parent_child.descendant_concept_id.isin(relation_additional_ids), 'label'] = 1



# number of child concepts
parent_child_counts = parent_child.groupby('ancestor_concept_id').agg(
    num_child=('descendant_concept_id', 'count'),
    num_child_relation=('label', 'sum'),
    similarity_model1=('similarity_model1', 'mean'),
    similarity_model2=('similarity_model2', 'mean')
).reset_index()

np.mean(parent_child_counts.similarity_model1)
# 0.45354092111632843
np.mean(parent_child_counts.similarity_model2)
# 0.5261579449977506


parent_child_filtered = parent_child_counts[
    (parent_child_counts.num_child_relation > 10)&
    (parent_child_counts.num_child > 20)&
    (parent_child_counts.num_child<50) &
    (parent_child_counts.similarity_model2 - parent_child_counts.similarity_model1 > 0.1)
    ].merge(
    target_concepts[['concept_id', 'concept_name']],
    left_on='ancestor_concept_id',
    right_on='concept_id',
    how='inner'
)[['concept_name', 'ancestor_concept_id', 'num_child', 'num_child_relation', 
  'similarity_model1', 'similarity_model2']]

parent_child_filtered.to_excel('report/JSM/parent_child_filtered.xlsx', index=False)
# id= 23731
# 81251

selected_parent = 4111017
selected_children = parent_child[parent_child.ancestor_concept_id == selected_parent].descendant_concept_id.unique()


target_concepts.concept_name[target_concepts.concept_id.isin(selected_children)].tolist()


# Create a 1-by-2 subplot
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

pca_plot_revised2(
    embeddings=target_concepts['model1_embedding'],
    highlight_ids=selected_children,
    target_concepts=target_concepts,
    ax=axes[0]
)


pca_plot_revised2(
    embeddings=target_concepts['model2_embedding'],
    highlight_ids=selected_children,
    target_concepts =target_concepts,
    ax=axes[1],
    ylabel=False
)

plt.savefig("report/JSM/PCA", bbox_inches='tight', dpi=300)
plt.close(fig)



#########################
# Test on the test dataset
#########################
from modules.FaissDB import build_index, is_initialized, delete_repository, search_similar
test_concept_pairs = pd.read_feather(os.path.join(matching_base_path, 'condition_matching_test_pos.feather'))

query_concepts = test_concept_pairs[['concept_id2', 'sentence2']].drop_duplicates()
query_concepts.to_excel('report/JSM/test_query_concepts.xlsx', index=False)
concept_ancestors = pd.read_feather(os.path.join(omop_base_path, 'concept_ancestor.feather'))

model1_emb = build_index(
    model = model1, 
    corpus_ids = target_concepts.concept_id, 
    corpus_names = target_concepts.concept_name, 
    corpus_embeddings = target_concepts['model1_embedding'],
    repos='model1'
)

model2_emb = build_index(
    model = model2,
    corpus_ids = target_concepts.concept_id,
    corpus_names = target_concepts.concept_name,
    corpus_embeddings = target_concepts['model2_embedding'],
    repos='model2'
)

top_k = 100
model1_top = search_similar(
    query_ids = query_concepts.concept_id2, 
    query_names = query_concepts.sentence2, 
    top_k=top_k, 
    repos='model1')

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

model1_eval = evaluate_performance(
    query_ids=model1_top['query_id'],
    similarities=model1_top['score'],
    labels=model1_top['label']
)

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