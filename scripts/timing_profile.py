import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import pandas as pd
import time
from tqdm import tqdm
import random

from modules.ML_data import DictBatchSampler, get_matching, get_relation
from datasets import IterableDataset, Value, Features

n_neg_matching = 4
n_neg_relation = 4
dt_seed = 42
data_folder = 'data/ML'
iterable_matching = get_matching(data_folder, n_neg=n_neg_matching, seed=dt_seed)
iterable_offspring, iterable_ancestor = get_relation(data_folder, n_neg = n_neg_relation, seed=dt_seed)



train_dataset = {
    'matching': iterable_matching,
    'offspring': iterable_offspring,
    'ancestor': iterable_ancestor
}

n_total = 100000
block_size = 1024
dict_ratio ={'matching': 0.5, 'offspring': 0.25, 'ancestor': 0.25}
# dict_ratio ={'matching': 1, 'offspring': 0, 'ancestor': 0}
sampler = DictBatchSampler(train_dataset, block_size, ratios = dict_ratio)

next(sampler.__iter__())

def myfunc():
    it = sampler.__iter__()
    for i in range(100):
        next(it)

import cProfile
import pstats

with cProfile.Profile() as pr:
    myfunc()
    
results = pstats.Stats(pr)
results.dump_stats('timing.prof')