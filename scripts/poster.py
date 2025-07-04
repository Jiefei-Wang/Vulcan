import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import numpy as np
import json
from modules.ML_train import auto_load_model
from modules.ChromaVecDB import ChromaVecDB
import pandas as pd
from modules.ChromaVecDB import ChromaVecDB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

model, tokenizer = auto_load_model('output/all-MiniLM-L6-v2_2025-04-23_12-21-04')

conceptEX         = pd.read_feather('data/omop_feather/conceptEX.feather')
conditions        = conceptEX[conceptEX['domain_id'] == 'Condition']
std_conditions    = conditions[conditions['standard_concept'] == 'S']
nonstd_conditions = conditions[conditions['standard_concept'] != 'S']
database = std_conditions[['concept_id', 'concept_name']]
query_df = nonstd_conditions[
    ['concept_id', 'concept_name', 'std_concept_id']
][nonstd_conditions['vocabulary_id'] == 'CIEL']


# query_df.to_excel('output/tmp/nonstd_conditions.xlsx', index=False)

db = ChromaVecDB(model=model, name='trained_eval', path=None)
db.empty_collection()
db.store_concepts(database, batch_size=5461)

max_k = 50
res = db.query(query_df[['concept_name']], n_results=max_k)

y_true = query_df['std_concept_id'].str[0].astype(int).tolist()


pred_lists = res['ids'] 

k_list = [1, 10, 50]

metrics = {}
for k in k_list:
    y_pred_top1 = [lst[0] if len(lst) > 0 else None for lst in pred_lists]
    hits = [1 if (y_true[i] in pred_lists[i][:k]) else 0 for i in range(len(y_true))]
    acc_k = sum(hits) / len(hits) 
    y_true_bin = [1] * len(y_true) 
    y_pred_bin = hits               
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average='binary', zero_division=0
    )
    metrics[f'Accuracy@{k}']  = acc_k
    metrics[f'Precision@{k}'] = p
    metrics[f'Recall@{k}']    = r
    metrics[f'F1@{k}']        = f1

print(json.dumps(metrics, indent=2, ensure_ascii=False))
"""
{
  "Accuracy@1": 0.8041032235438524,
  "Precision@1": 1.0,
  "Recall@1": 0.8041032235438524,
  "F1@1": 0.8914159822455492,
  "Accuracy@10": 0.8624075751819271,
  "Precision@10": 1.0,
  "Recall@10": 0.8624075751819271,
  "F1@10": 0.9261212064149641,
  "Accuracy@50": 0.877809276090832,
  "Precision@50": 1.0,
  "Recall@50": 0.877809276090832,
  "F1@50": 0.9349291083684809
}
"""

reciprocal_ranks = []
for true_id, preds in zip(y_true, pred_lists):
    try:
        rank = preds.index(true_id) + 1
        reciprocal_ranks.append(1.0 / rank)
    except ValueError:
        reciprocal_ranks.append(0.0)

mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
"""
Mean Reciprocal Rank (MRR): 0.8258
"""
# the correct answer to appear around first or second position most of the time.