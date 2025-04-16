import os
import pandas as pd
import numpy as np
from modules.ML_sampling import AncestorIterableDataset, MatchingIterableDataset, OffspringIterableDataset
import itertools
import random
from datasets import IterableDataset, Dataset
from tqdm import tqdm


def get_matching(data_folder, n_neg=4, seed=42):
    positive_dataset_matching = pd.read_feather(
        os.path.join(data_folder, 'matching/positive_dataset_matching.feather')
        )
    candidate_df_matching = pd.read_feather(
        os.path.join(data_folder, 'matching/candidate_dataset_matching.feather')
        )
    
    candidate_fp = pd.read_feather(
        os.path.join(data_folder, 'matching/candidate_fp.feather')
        )
    

    iterable_matching = MatchingIterableDataset(
        positive_df_matching = positive_dataset_matching,
        candidate_df_matching = candidate_df_matching,
        candidate_fp_matching = candidate_fp,
        n_neg=n_neg,  
        seed=seed
    )
    
    return iterable_matching


def get_relation(data_folder, n_neg=4, seed=42):
    positive_dataset_relation = pd.read_feather(
        os.path.join(data_folder, 'relation/positive_dataset_relation.feather')
        )
    std_target = pd.read_feather(
        os.path.join(data_folder, 'relation/std_target.feather')
        )
    
    
    
    iterable_offspring = OffspringIterableDataset(
        positive_dataset_relation,
        std_target,
        n_neg=n_neg,
        seed=seed
    )
    
    iterable_ancestor = AncestorIterableDataset(
        positive_dataset_relation,
        std_target,
        n_neg=n_neg,
        seed=seed
    )
    
    return iterable_offspring, iterable_ancestor

def get_matching_validation(data_folder, n_neg=1, seed=42):
    positive_matching_validation = pd.read_feather(
        os.path.join(data_folder, 'validation/positive_matching_validation.feather')
        )
    
    candidate_matching_validation = pd.read_feather(
        os.path.join(data_folder, 'validation/candidate_matching_validation.feather')
        )
    
    iterable_matching_validation = MatchingIterableDataset(
        positive_matching_validation,
        candidate_matching_validation,
        n_neg=n_neg,  
        seed=seed,
        special_token_sentence1=True,
        special_token_sentence2=False,
        special_token_candidate=False
    )
    return iterable_matching_validation


def get_relation_positive_validation(data_folder):
    positive_dataset_to_offspring_validation = pd.read_feather(
        os.path.join(data_folder, 'validation/positive_dataset_to_offspring_validation.feather')
        )
        
    positive_dataset_to_ancestor_validation = pd.read_feather(
        os.path.join(data_folder, 'validation/positive_dataset_to_ancestor_validation.feather')
        )
    
    return positive_dataset_to_offspring_validation, positive_dataset_to_ancestor_validation


import random
from collections import deque

def buffered_shuffle(rng, iterator, buffer_size):
    buffer = deque()

    # Fill the buffer initially
    try:
        for _ in tqdm(range(buffer_size)):
            buffer.append(next(iterator))
    except StopIteration:
        rng.shuffle(buffer)
        yield from buffer
        return

    for item in iterator:
        idx = rng.randint(0, buffer_size - 1)
        yield buffer[idx]
        buffer[idx] = item

    # Shuffle and yield the remaining buffer
    rng.shuffle(buffer)
    yield from buffer


def cycle_from_generator(generator_func):
    while True:
        for item in generator_func():
            yield item


class DictBatchSampler:
    def __init__(self, datasets_dict, batch_size, ratios=None, shuffle = True, seed=42, shuffle_buffer=8*1024):
        """
        datasets_dict: dict of {'name': IterableDataset}
        batch_size: total number of samples per batch
        ratios: dict specifying ratios for each dataset {'name': float}. If None, equal ratios assumed.
        seed: random seed for shuffling
        """
        self.rng = random.Random(seed)  # Create a separate random stream
        self.datasets_dict = datasets_dict
        self.batch_size = batch_size
        self.names = list(datasets_dict.keys())

        # Use cycle() to avoid frequent StopIteration handling
        self.cyncled_iter = {k: cycle_from_generator(v.trainer_iter) for k, v in self.datasets_dict.items()}
        
        if shuffle:
            self.shuffled_iter = {k:buffered_shuffle(self.rng, v, shuffle_buffer) for k, v in self.cyncled_iter.items()}
        else:
            self.shuffled_iter = self.cyncled_iter
        
        self.iterators = self.shuffled_iter
        
        if ratios:
            if set(ratios.keys()) != set(self.names):
                raise ValueError("Ratios keys must match dataset names")
            total = sum(ratios.values())
            self.num_samples = {k: int(ratios[k]/total*batch_size) for k in self.names}
        else:
            self.num_samples = {k: int(1/len(self.names)*batch_size) for k in self.names}

        # Adjust num_samples to match batch_size exactly
        current_total = sum(self.num_samples.values())
        if current_total != batch_size:
            diff = batch_size - current_total
            for i, k in enumerate(self.names[:diff]):
                self.num_samples[k] += 1
                

    def __iter__(self):
        return self

    def element_size(self):
        """
        Return the number of elements in the dataset
        """
        try:
            samples = {k: v.element_size() for k,v in self.datasets_dict.items()}
        except TypeError:
            raise ValueError("All datasets must support element_size() to calculate iteration size")
        return samples
    
    def iteration_size(self):
        """
        Return the theoretical minimum number of iterations (batches) needed to process all datasets once,
        based on their lengths and sampling ratios. 
        Raises ValueError if dataset lengths are unavailable
        """

        # Check for zero in num_samples
        if 0 in self.num_samples.values():
            raise ValueError("num_samples contains zero values, cannot compute iterations")

        # Check if lengths are available
        samples = self.element_size()
        
        iter_len = {}
        for k, s in samples.items():
            ns = self.num_samples[k]
            if ns == 0:
                iter_len[k] = 0
            else: 
                iter_len[k] = int(np.ceil(s / ns))
            
        return iter_len

    def __next__(self):
        batch = []
        for name in self.names:
            num_samples = self.num_samples[name]
            ## do nothing if there is no sample to take
            if num_samples == 0:
                continue
            batch.extend(itertools.islice(self.iterators[name], num_samples))

        self.rng.shuffle(batch)  # Shuffle using the separate random stream
        batch_ds = Dataset.from_list(batch)
        return batch_ds  # Return a list directly, avoid Dataset.from_list overhead