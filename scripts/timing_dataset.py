import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import pandas as pd
import time
from tqdm import tqdm

from modules.ML_data import DictBatchSampler, get_matching, get_relation
from datasets import IterableDataset, Value, Features

data_folder = 'data/ML'


# positive_dataset_matching = pd.read_feather('data/ML/matching/positive_dataset_matching.feather')

# iterable_matching = get_matching(data_folder)
# len(iterable_matching)

iterable_matching = get_matching(data_folder, n_fp=4)
len(iterable_matching)

def measuring_time(it, ntotal=None, n_limit=400000):
    ## if len(it) is defined, use it
    if ntotal is None and hasattr(it, 'len'):
        ntotal = it.len()
    n=0
    start = time.time()
    for i in tqdm(it, total=ntotal):
        n += 1
        if n > n_limit:
            break
    end = time.time()
    return end - start




measuring_time(iterable_matching)
# 4.76409125328064


n_neg_relation = 4
dt_seed = 42
iterable_offspring, iterable_ancestor = get_relation(data_folder, n_neg = n_neg_relation, seed=dt_seed)

iterable_offspring.num_candidates

measuring_time(iterable_offspring)
# 6.8772056102752686
measuring_time(iterable_ancestor)
# 5.3391149044036865



## Sampler
n_fp = 4
n_neg_matching = 4
n_neg_relation = 4
dt_seed = 42
data_folder = 'data/ML'
iterable_matching = get_matching(data_folder, n_neg=n_neg_matching, seed=dt_seed, n_fp = n_fp)
iterable_offspring, iterable_ancestor = get_relation(data_folder, n_neg = n_neg_relation, seed=dt_seed)


len(iterable_matching)
len(iterable_offspring)
len(iterable_ancestor)

it = iterable_matching.trainer_iter()
next(it)

it = iter(iterable_matching)
next(it)

it = iterable_offspring.trainer_iter()
next(it)

it = iter(iterable_ancestor)
next(it)



train_dataset = {
    'matching': iterable_matching,
    # 'offspring': iterable_offspring,
    'ancestor': iterable_ancestor
}




n_total = 100000
block_size = 1024
dict_ratio ={'matching': 2, 'ancestor': 1}
# dict_ratio ={'matching': 1, 'offspring': 0, 'ancestor': 0}
sampler = DictBatchSampler(train_dataset, block_size, ratios = dict_ratio, train=False)

it = iter(sampler)
dt = next(it)
# df = pd.DataFrame(dt)
# print(df.head(10))
# import xlwings as xw
# xw.view(df)

measuring_time(sampler, n_total/block_size, n_limit = n_total/block_size)
measuring_time(iterable_matching, n_total, n_limit = n_total)
# measuring_time(matching_ds, n_total, n_limit = n_total)
measuring_time(iterable_offspring, n_total, n_limit = n_total)
# measuring_time(offspring_ds, n_total, n_limit = n_total)
measuring_time(iterable_ancestor, n_total, n_limit = n_total)
# measuring_time(ancestor_ds, n_total, n_limit = n_total)


# n_total = 100000
# block_size = 1024
# >>> measuring_time(sampler, n_total/block_size, n_limit = n_total/block_size)
#  99%|███████████████████████████████████████████████████████▌| 97/97.65625 [00:05<00:00, 18.25it/s]
# 41.37319350242615
# 4.990448713302612
# >>> measuring_time(iterable_matching, n_total, n_limit = n_total)
# 100%|███████████████████████████████████████████████████| 100000/100000 [00:01<00:00, 57017.17it/s]
# 70.65507745742798
# 1.8288779258728027
# >>> measuring_time(iterable_offspring, n_total, n_limit = n_total)
# 100%|███████████████████████████████████████████████████| 100000/100000 [00:08<00:00, 12228.59it/s] 
# 8.17855954170227
# 8.840962648391724
# >>> measuring_time(iterable_ancestor, n_total, n_limit = n_total)
# 100%|███████████████████████████████████████████████████| 100000/100000 [00:07<00:00, 12653.94it/s] 
# 7.902676582336426
# 1.4610817432403564




positive_df = positive_dataset_matching

positive_df=positive_df.reset_index(drop=True)
positive_df['index'] = positive_df.index
positive_df = positive_df.set_index('concept_id1')


positive_df.loc[42600379,"index"]["index"].values

positive_df.iloc[0]
positive_df[42600379]