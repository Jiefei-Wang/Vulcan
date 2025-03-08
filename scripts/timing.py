import pandas as pd
import time
from tqdm import tqdm
import random

from modules.ML_data import get_matching
from datasets import IterableDataset


positive_dataset_matching = pd.read_feather('data/ML/matching/positive_dataset_matching.feather')

data_folder = 'data/ML'
iterable_matching = get_matching(data_folder)
matching_ds = IterableDataset.from_generator(iterable_matching.__iter__)

def measuring_time(it, ntotal, n_limit=100000):
    n=0
    start = time.time()
    for i in tqdm(it, total=ntotal):
        n += 1
        if n > n_limit:
            break
    end = time.time()
    return end1 - start1



ntotal = len(positive_dataset_matching)*5
ntotal

measuring_time(iterable_matching, ntotal)
# 5.217

measuring_time(matching_ds, ntotal)
# 5.217

it = matching_ds.shuffle(seed=42)
measuring_time(it, ntotal)
# 5.217


