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





base_model = 'all-MiniLM-L6-v2'
base_model_path = f'models/{base_model}'
trained_model_path = "output/all-MiniLM-L6-v2_2025-03-31_13-25-49/best_model"

special_tokens = ['[MATCHING]', '[OFFSPRINT]', '[ANCESTOR]']


model_base = SentenceTransformer(base_model_path)
model_train = SentenceTransformer(trained_model_path)


####################################
## Work on conditions domain
## CIM vocabulary maps to standard concepts
####################################

# Read dataset
conceptEX = pd.read_feather('data/omop_feather/conceptEX.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')

conditions = conceptEX[conceptEX['domain_id'] == 'Condition']

std_conditions = conditions[conditions['standard_concept'] == 'S']
nonstd_conditions = conditions[conditions['standard_concept'] != 'S']

# conditions.columns
# conditions.vocabulary_id.unique()
# std_conditions.vocabulary_id.unique()
# nonstd_conditions.vocabulary_id.unique()


database = std_conditions[['concept_id', 'concept_name']]
query = nonstd_conditions[['concept_id', 'concept_name', 'std_concept_id']][nonstd_conditions['vocabulary_id'] == 'CIEL']


query_match = query.copy()
query_match['concept_name'] = add_special_token(query_match['concept_name'], '[MATCHING]')




validation_set = [
    {
        'name' : 'base',
        'model': model_base,
        'database': database,
        'query': query,
    },
    # {
    #     'name' : 'matching_only',
    #     'model': 'models/base_no_relation/final',
    #     'std': std_data_token,
    #     'nonstd': nonstd_data_token,
    # },
    {
        'name' : 'matching_and_relation',
        'model': model_train,
        'database': database,
        'query': query_match,
    }
]

## TODO: Create a single db
## use model + concept to do embedding

return_results = 100
k_list= [1, 10, 50]
result = []

# path = tempfile.mkdtemp()
path = None
for i in range(len(validation_set)):
    value = validation_set[i]
    name = value['name']
    model = value['model']
    database_i = value['database']
    query_i = value['query']
    db = ChromaVecDB(model=model, name=name, path=path)
    db.empty_collection()
    db.store_concepts(database_i, batch_size= 5461)
    
    df_test = map_concepts(db, query_i, n_results=return_results)
    res = {str(k): performance_metrics(df_test,k=k) for k in k_list}
    res['model'] = name
    result.append(res)
    

result


print(json.dumps(result, indent=2))
# [
#   {
#     "1": 0.7462664757284391,
#     "10": 0.823830259812374,
#     "50": 0.8491393167139142,
#     "model": "models/all-MiniLM-L6-v2"
#   },
#   {
#     "1": 0.7605283923196072,
#     "10": 0.8459537656720344,
#     "50": 0.8736592921647135,
#     "model": "models/base_no_relation/final"
#   },
#   {
#     "1": 0.74848759388608,
#     "10": 0.8339129672385072,
#     "50": 0.8666452348247947,
#     "model": "models/base_exclude_CIEL/final"
#   }
# ]