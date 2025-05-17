import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import pandas as pd
from modules.ML_train import auto_load_model
from modules.ChromaVecDB import ChromaVecDB



model, tokenizer = auto_load_model('output/all-MiniLM-L6-v2_2025-04-23_12-21-04')



conceptEX = pd.read_feather('data/omop_feather/conceptEX.feather')
conditions = conceptEX[conceptEX['domain_id'] == 'Condition']

std_conditions = conditions[conditions['standard_concept'] == 'S']
nonstd_conditions = conditions[conditions['standard_concept'] != 'S']

database = std_conditions[['concept_id', 'concept_name']]
query = nonstd_conditions[['concept_id', 'concept_name', 'std_concept_id']][nonstd_conditions['vocabulary_id'] == 'CIEL']


db = ChromaVecDB(model=model, name="test")
db.empty_collection()
db.store_concepts(database, batch_size= 5461)
    

query_df = pd.DataFrame({'concept_name': ["changes in bowel habits"]})

res = db.query(query_df, n_results=10)

std_conditions[std_conditions['concept_id'].isin(res['ids'][0])]
