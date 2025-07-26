from modules.FaissDB import delete_repository
from modules.FalsePositives import get_false_positives
import numpy as np
import pandas as pd
from torch.utils.data import Dataset





class PositiveDataset(Dataset):
    column_names = ['sentence1', 'sentence2', 'label']
    def __init__(self, target_concepts, name_table, name_bridge, max_elements=10, label = 1, seed=None):
        self.target_concepts = target_concepts.copy()
        self.name_table = name_table.copy()
        self.name_bridge = name_bridge.copy()
        self.max_elements = max_elements
        self.label = label
        
        self.concept_id_2_name = {i:v for i, v in zip(self.target_concepts['concept_id'], self.target_concepts['concept_name'])}
        self.name_id_2_name = {i:v for i, v in zip(self.name_table['name_id'], self.name_table['name'])}
        
        self.resample(seed)
        
    def resample(self, seed=None):
        """
        Resample the dataset to give each concept_id has randomly selected max_elements entries.
        """
        filtered_bridge = self.name_bridge[self.name_bridge['concept_id'].isin(self.target_concepts['concept_id'])]
        # for bridge, group by concept_id and only keep the first max_elements entries
        if seed is None:
            filtered_bridge = filtered_bridge.groupby('concept_id').head(self.max_elements)
        else:
            filtered_bridge = filtered_bridge.sample(frac=1, random_state=seed).groupby('concept_id').head(self.max_elements)
        self.filtered_bridge = filtered_bridge.reset_index(drop=True)
        self.seed = seed
        return self
        
    def __len__(self):
        return len(self.filtered_bridge)

    def __getitem__(self, idx):
        matching = self.filtered_bridge.iloc[idx]
        concept_id1 = matching['concept_id']
        concept_id2 = matching['name_id']
        sentence1 = self.concept_id_2_name[concept_id1]
        sentence2 = self.name_id_2_name[concept_id2]
        
        return {"sentence1": sentence1, "sentence2": sentence2, "label": self.label}
    
    def __str__(self):
        ## show the total length
        return f"PositiveDataset(length={len(self)}, label={self.label}, seed={self.seed})"
    
    def __repr__(self):
        return self.__str__()

class NegativeDataset(Dataset):
    column_names = ['sentence1', 'sentence2', 'label']
    def __init__(self, target_concepts, name_table, blacklist_bridge, max_elements=10, seed=42):
        self.target_concepts = target_concepts.copy()
        self.name_table = name_table.copy()
        self.blacklist_bridge = blacklist_bridge.copy()
        self.max_elements = max_elements
        self.all_concept_ids = target_concepts['concept_id'].values
        self.all_name_ids = name_table['name_id'].values
        
        self.concept_id_2_name = {i:v for i, v in zip(self.target_concepts['concept_id'], self.target_concepts['concept_name'])}
        
        self.name_id_2_name = {i:v for i, v in zip(self.name_table['name_id'], self.name_table['name'])}
        
        # Create blacklist map: concept_id -> set of blacklisted name_ids
        self.blacklist_map = blacklist_bridge.groupby('concept_id')['name_id'].apply(set).to_dict()
        self.resample(seed)
        
    def resample(self, seed=None):
        """
        Resample the dataset to give each concept_id has randomly selected max_elements entries.
        """
        if seed is None:
            seed = 0
        rng = np.random.default_rng(seed)
        bridge = pd.DataFrame({
            'concept_id': np.repeat(self.all_concept_ids, self.max_elements),
            'name_id': rng.choice(self.all_name_ids, size=len(self.all_concept_ids) * self.max_elements, replace=True)
        }).drop_duplicates()
        bridge = bridge.set_index(['concept_id', 'name_id'])
        blacklist_bridge = self.blacklist_bridge.set_index(['concept_id', 'name_id'])
        bridge = bridge[~bridge.index.isin(blacklist_bridge.index)]
        
        bridge.reset_index(drop=False, inplace=True)
        bridge.set_index('concept_id', inplace=True)
        counts = bridge.groupby('concept_id')['name_id'].count()
        # group by concept_id and find those that have less than max_elements entries
        missing_counts = counts[counts < self.max_elements]
        
        # for those concept_ids, find the missing name_ids and add them to the bridge
        new_list = []
        for concept_id, count in zip(missing_counts.index, missing_counts.values):
            result = bridge.loc[concept_id]
            if isinstance(result, pd.Series):
                # Single match - convert Series to list
                selected = [result['name_id']]
            else:
                # Multiple matches - extract column and convert to list
                selected = result['name_id'].tolist()
            num_sample = self.max_elements - len(selected)
            if num_sample > 0:
                new_samples = self._get_random_sample(rng, concept_id, selected, num_sample)
                for new_sample in new_samples:
                    new_list.append({'concept_id': concept_id, 'name_id': new_sample})
        
        bridge.reset_index(inplace=True)
        if new_list:
            new_samples_df = pd.DataFrame(new_list)
            bridge = pd.concat([bridge, new_samples_df])
        
        bridge = bridge.groupby('concept_id').head(self.max_elements)
        self.bridge = bridge
        self.seed = seed
        return self
        
    def _get_random_sample(self, rng, concept_id, selected, num_sample):
        black_list = list(self.blacklist_map.get(concept_id, set()) | set(selected))
        candidates = np.setdiff1d(self.all_name_ids, black_list, assume_unique=True)
        num_sample = min(num_sample, len(candidates))
        if num_sample > 0:
            return rng.choice(candidates, size=num_sample, replace=False)
        return []
        
        
    def __len__(self):
        return len(self.bridge)

    def __getitem__(self, idx):
        matching = self.bridge.iloc[idx]
        concept_id1 = matching['concept_id']
        concept_id2 = matching['name_id']
        sentence1 = self.concept_id_2_name[concept_id1]
        sentence2 = self.name_id_2_name[concept_id2]
        return {"sentence1": sentence1, "sentence2": sentence2, "label": 0}

    def __str__(self):
        ## show the total length
        return f"PositiveDataset(length={len(self)}, seed={self.seed})"
    
    def __repr__(self):
        return self.__str__()


class FalsePositiveDataset():
    def __init__(self, target_concepts, n_fp_matching = 50, existing_path=None):
        if existing_path is not None:
            fp_matching = pd.read_feather(existing_path)
            fp_matching = fp_matching[['sentence1', 'sentence2', 'label']].copy()
            self.fp_matching = fp_matching
        self.target_concepts = target_concepts.copy()
        self.n_fp_matching = n_fp_matching
        
    def add_model(self, model):
        self.model = model
    
    def resample(self, seed=None):
        target_concepts = self.target_concepts
        model = self.model
        n_fp_matching = self.n_fp_matching
        
        delete_repository(repos='training_false_positive')
        fp_matching = get_false_positives(
            model=model,
            corpus_concepts=target_concepts,
            n_fp=n_fp_matching,
            repos='training_false_positive'
        )
        self.fp_matching = fp_matching[['sentence1', 'sentence2', 'label']].copy()
        return self
    
    def __len__(self):
        return len(self.fp_matching)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slice objects
            return self.fp_matching.iloc[idx].to_dict(orient='records')
        else:
            return self.fp_matching.iloc[idx].to_dict()
    
    def __str__(self):
        return f"FalsePositiveDataset(length={len(self)}, n_fp_matching={self.n_fp_matching})"
    
    def __repr__(self):
        return self.__str__()
        


class CombinedDataset():
    def __init__(self, **kwargs):
        """
        Combines multiple datasets into a single dataset.
        
        Args:
            **kwargs: Datasets to combine, where keys are dataset names and values are dataset objects. The dataset can be pd.DataFrame or any other object that supports [] indexing.
        """
        self.datasets = kwargs
        self.names = list(self.datasets.keys())
        self.lengths = [len(self.datasets[name]) for name in self.names]
        self.partial_lengths = np.cumsum(self.lengths)
        self.total_length = sum(self.lengths)
        self.shuffle()
        
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            # Handle slice objects
            start, stop, step = key.indices(self.total_length)
            return [self._get_single_item(i) for i in range(start, stop, step)]
        else:
            # Handle single index
            return self._get_single_item(key)
    
    def _get_single_item(self, idx):
        _idx = self.index_mapping[idx]
        if _idx < 0 or _idx >= self.total_length:
            raise IndexError("Index out of range")
            
        for i, partial_len in enumerate(self.partial_lengths):
            if _idx < partial_len:
                dataset_name = self.names[i]
                dataset = self.datasets[dataset_name]
                previous_len = self.partial_lengths[i-1] if i > 0 else 0
                local_idx = _idx - previous_len
                return dataset.iloc[local_idx].to_dict()  if isinstance(dataset, pd.DataFrame) else dataset[local_idx]
        raise IndexError("Index out of range")
    
    def shuffle(self, seed=None):
        """
        Shuffle the dataset indices based on the provided seed. If no seed is provided, the dataset will be restored to its original order.
        """
        if seed is None:
            self.index_mapping = np.arange(self.total_length)
        else:
            # for name, dataset in self.datasets.items():
            #     if isinstance(dataset, PositiveDataset) or isinstance(dataset, NegativeDataset):
            #         dataset = dataset.shuffle(seed)
            #         self.datasets[name] = dataset
            self.index_mapping = np.random.default_rng(seed).permutation(self.total_length)
        
        self.seed = seed
        return self
    
    def resample(self, seed=None):
        for name, dataset in self.datasets.items():
            ## check if it has a resample method
            ## for data.frame, we just ignore it
            if not isinstance(dataset, pd.DataFrame) and hasattr(dataset, 'resample'):
                dataset = dataset.resample(seed)
                self.datasets[name] = dataset
        
        self.lengths = [len(self.datasets[name]) for name in self.names]
        self.partial_lengths = np.cumsum(self.lengths)
        self.total_length = sum(self.lengths)
        return self    
    
    def __str__(self):
        return f"CombinedDataset(length={len(self)}, datasets={self.names}, seed={self.seed})"
    
    def __repr__(self):
        return self.__str__()