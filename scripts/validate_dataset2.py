import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import pandas as pd
import time
from tqdm import tqdm

from modules.ML_data import get_matching, get_relation


n_fp = 4
n_neg_matching = 4
n_neg_relation = 4
dt_seed = 42
data_folder = 'data/ML'
iterable_matching = get_matching(data_folder, n_neg=n_neg_matching, seed=dt_seed, n_fp = n_fp)
iterable_offspring, iterable_ancestor = get_relation(data_folder, n_neg = n_neg_relation, seed=dt_seed)


n_positive = 0
n_negative = 0
n_false_positive = 0
n_total = len(iterable_matching)
for example in tqdm(iter(iterable_matching), total=len(iterable_matching)):
    is_pos = example['from'] == "positive"
    is_neg = example['from'] == "negative"
    is_fp = example['from'] == "false_positive"
    
    
    n_positive += is_pos
    n_negative += is_neg
    n_false_positive += is_fp
    
    if is_pos:
        assert example['label'] == 1, f"Positive sample should have label 1, but got {example['label']}"
        
    if is_neg:
        assert example['label'] == 0, f"Negative sample should have label 0, but got {example['label']}"
        
    if is_fp:
        assert example['label'] == 0, f"False positive sample should have label 0, but got {example['label']}"



n_positive, n_negative, n_false_positive, n_total
