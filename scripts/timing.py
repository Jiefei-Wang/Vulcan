import pandas as pd
import time
from tqdm import tqdm
import random

from modules.ML_sampling import MatchingIterableDataset, RelationNegativeIterableDataset, add_special_token

positive_dataset_matching = pd.read_feather('data/ML/matching/positive_dataset_matching.feather')
candidate_df_matching = pd.read_feather('data/ML/matching/candidate_dataset_matching.feather')


iterable_matching = MatchingIterableDataset(
    positive_dataset_matching,
    candidate_df_matching,
    n_neg=4,  
    seed=42
)

len(positive_df_matching)*5

n=0
start1 = time.time()
for it in tqdm(iterable_matching):
    n += 1
    # if n > 1000000:
    #     break
end1 = time.time()
print(end1 - start1)

print(iterable_matching)
# 43.80s


