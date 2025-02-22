import pandas as pd
import time
from tqdm import tqdm
import random

from modules.ML_sampling import MatchingIterableDataset


concept = pd.read_feather('data/omop_feather/concept.feather')
concept_ancestor = pd.read_feather('data/omop_feather/concept_ancestor.feather')
        
positive_dataset_matching = pd.read_feather('data/ML/matching/positive_dataset_matching.feather')
candidate_df_matching = pd.read_feather('data/ML/matching/candidate_dataset_matching.feather')


iterable_matching = MatchingIterableDataset(
    positive_dataset_matching,
    candidate_df_matching,
    n_neg=4,  
    seed=42
)

blacklist = positive_dataset_matching.groupby('sentence1')["sentence2"].apply(set).to_dict()

## Check sample size
label0= 0
label1= 0
for it in tqdm(iterable_matching):
    ## TODO: check if concept_id matchs concept name
    sentence1 = it['sentence1']
    sentence2 = it['sentence2']
    label = it['label']
    bl = blacklist.get(sentence1)
    ## check if sentence2 has a [MATCHING] label in the beginning
    assert sentence2.startswith("[MATCHING]")
    sentence2 = sentence2[11:]
    
    if label == 1:
        assert sentence2 in bl
        label1 += 1
    else:
        ## TODO: Some concept names are the same, one in the blacklist and one not, what should we do?
        assert sentence2 not in bl
        label0 += 1

label0
label1
