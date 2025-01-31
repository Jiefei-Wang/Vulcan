## For Windows: Python version 10, chromaDB version 0.5.4
import pandas as pd
from sentence_transformers import SentenceTransformer
from modules.ChromaVecDB import ChromaVecDB
from modules.performance import map_concepts, performance_metrics
import tempfile


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

nonstd_cim_conditions = nonstd_conditions[nonstd_conditions['vocabulary_id'] == 'CIM10']
nonstd_cim_conditions.iloc[0]

model_names = ['models/all-MiniLM-L6-v2', 'models/fine-tuned/checkpoint-46530']




####################################
## Generate embeddings for standard and non-standard concepts
####################################
model1 = SentenceTransformer('models/all-MiniLM-L6-v2')
model2 = SentenceTransformer('models/fine-tuned/checkpoint-46530')
model3 = SentenceTransformer('models/fine-tuned2/final')

####################################
## chromadb for fast similarity search
## Only store standard condition embeddings
####################################
## create system temp directory

path1 = tempfile.mkdtemp()
db1 = ChromaVecDB(model=model1, doc_name="ref", path=path1)
db1.store_concepts(std_conditions)

path2 = tempfile.mkdtemp()
db2 = ChromaVecDB(model=model2, doc_name="finetuned", path=path2)
db2.store_concepts(std_conditions)


path3 = tempfile.mkdtemp()
db3 = ChromaVecDB(model=model3, doc_name="finetuned", path=path3)
db3.store_concepts(std_conditions)

####################################
## Find if the standard concept id is in the top k
## for each non-standard concept
####################################
n_results = 20
df_test1 = map_concepts(db1, nonstd_cim_conditions, n_results=n_results)
performance_metrics(df_test1,k=1)
performance_metrics(df_test1,k=10)
performance_metrics(df_test1,k=50)

df_test2 = map_concepts(db2, nonstd_cim_conditions, n_results=n_results)
performance_metrics(df_test2,k=1)
performance_metrics(df_test2,k=10)
performance_metrics(df_test2,k=50)


df_test3 = map_concepts(db3, nonstd_cim_conditions, n_results=n_results)
performance_metrics(df_test3,k=1)
performance_metrics(df_test3,k=10)