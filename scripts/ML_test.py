## For Windows: Python version 10, chromaDB version 0.5.4
import pandas as pd
from sentence_transformers import SentenceTransformer
from modules.ChromaVecDB import ChromaVecDB
from modules.performance import map_concepts, performance_metrics
import tempfile
from modules.ML_sampling import add_special_token
import json

# Read dataset
conceptEX = pd.read_feather('data/omop_feather/conceptEX.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')

####################################
## Work on conditions domain
## CIM vocabulary maps to standard concepts
####################################
conditions = conceptEX[conceptEX['domain_id'] == 'Condition']

std_conditions = conditions[conditions['standard_concept'] == 'S']
nonstd_conditions = conditions[conditions['standard_concept'] != 'S']

# conditions.columns
# conditions.vocabulary_id.unique()
# std_conditions.vocabulary_id.unique()
# nonstd_conditions.vocabulary_id.unique()


std_data = std_conditions[['concept_id', 'concept_name']]
nonstd_data = nonstd_conditions[['concept_id', 'concept_name', 'std_concept_id']][nonstd_conditions['vocabulary_id'] == 'CIEL']

std_data_token = std_data.copy()
std_data_token['concept_name'] = add_special_token(std_data_token['concept_name'], '[MATCHING]')

nonstd_data_token = nonstd_data.copy()
nonstd_data_token['concept_name'] = add_special_token(nonstd_data_token['concept_name'], '[MATCHING]')



validation_set = [
    {
        'name' : 'base',
        'model': 'models/all-MiniLM-L6-v2',
        'std': std_data,
        'nonstd': nonstd_data,
    },
    {
        'name' : 'matching_only',
        'model': 'models/base_no_relation/final',
        'std': std_data_token,
        'nonstd': nonstd_data_token,
    },
    {
        'name' : 'matching_and_relation',
        'model': 'models/base_exclude_CIEL/final',
        'std': std_data_token,
        'nonstd': nonstd_data_token,
    }
]

## TODO: Create a single db
## use model + concept to do embedding

return_results = 100
k_list= [1, 10, 50]
result = []

path = tempfile.mkdtemp()
path = None
for i in range(len(validation_set)):
    value = validation_set[i]
    name = value['name']
    model_name = value['model']
    std = value['std']
    nonstd = value['nonstd']
    model = SentenceTransformer(model_name)
    db = ChromaVecDB(model=model, name=name, path=path)
    db.empty_collection()
    db.store_concepts(std)
    
    df_test = map_concepts(db, nonstd, n_results=return_results)
    res = {str(k): performance_metrics(df_test,k=k) for k in k_list}
    res['model'] = model_name
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