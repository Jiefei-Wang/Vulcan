import numpy as np
import pandas as pd

class PositiveIterable:
    def __init__(self, 
                 target_concepts,
                 name_table,
                 name_bridge,
                 max_element = 999999
                 ):
        """
        iterable dataset for positive samples, label is 1
        
        Args:
            target_concepts (pd.DataFrame): DataFrame containing target concepts with columns ['concept_id', 'concept_name']
            name_table (pd.DataFrame): DataFrame containing name mappings with columns ['name_id', 'name']
            name_bridge (pd.DataFrame): DataFrame containing positive samples with columns ['concept_id', 'name_id']
            max_element (int, optional): The maximum number of elements to return for each concept in target_concepts. Defaults to 999999.
            seed (int): Random seed for reproducibility.
        """
        self.name_bridge = name_bridge
        self.max_element = max_element
        
        self.target_concepts = target_concepts[target_concepts['concept_id'].isin(name_bridge['concept_id'])]
        
        # Pre-compute concept name lookup
        self.concept_name_lookup = dict(zip(target_concepts['concept_id'], target_concepts['concept_name']))
        
        # Pre-compute name_id to name info lookup
        self.name_lookup = dict(zip(name_table['name_id'], name_table['name']))
        
        # Pre-compute grouped name_bridge by concept_id for efficient iteration
        self.concept_groups = {}
        for concept_id, group in self.name_bridge.groupby('concept_id'):
            if concept_id in self.concept_name_lookup:
                self.concept_groups[concept_id] = group.head(self.max_element)
    
    def __iter__(self):
        for concept_id, group in self.concept_groups.items():
            concept_name = self.concept_name_lookup[concept_id]
            
            for _, row in group.iterrows():
                name_id = row['name_id']
                
                # Skip if name_id not found in name_table
                if name_id not in self.name_lookup:
                    continue
                    
                name = self.name_lookup[name_id]
                
                yield {
                    'sentence1': concept_name,
                    'sentence2': name,
                    'concept_id1': concept_id,
                    'concept_id2': name_id,
                    'label': 1
                }


class FalsePositiveIterable(PositiveIterable):
    def __init__(self, 
                 target_concepts,
                 name_table,
                 name_bridge,
                 max_element = 999999
                 ):
        """
        iterable dataset for false positive samples, label is 0
        """
        super().__init__(target_concepts, name_table, name_bridge, max_element)
    
    def __iter__(self):
        for item in super().__iter__():
            item['label'] = 0
            yield item

class NegativeIterable:
    def __init__(self, 
                 target_concepts,
                 name_table,
                 blacklist_name_bridge,
                 max_element = 5,
                 seed = 42
                 ):
        """
        iterable dataset for negative samples, label is 0
        
        Args:
            target_concepts (pd.DataFrame): DataFrame containing target concepts with columns ['concept_id', 'concept_name']
            name_table (pd.DataFrame): DataFrame containing name mappings with columns ['name_id', 'name']
            blacklist_name_bridge (pd.DataFrame): DataFrame containing samples that cannot be paired ['concept_id', 'name_id']
            max_element (int, optional): The maximum number of elements to return for each concept in target_concepts. Defaults to 5.
            seed (int): Random seed for reproducibility.
        """
        self.max_element = max_element
        self.seed = seed
        self.target_concept_ids = target_concepts['concept_id'].values
        self.all_name_ids = name_table['name_id'].values
        self.num_candidates = len(self.all_name_ids)
        
        # Pre-compute concept name lookup
        self.concept_name_lookup = dict(zip(target_concepts['concept_id'], target_concepts['concept_name']))
        
        # Pre-compute name_id to name info lookup
        self.name_lookup = dict(zip(name_table['name_id'], name_table['name']))
        
        # Create blacklist map: concept_id -> set of blacklisted name_ids
        blacklist_grouped = blacklist_name_bridge.groupby('concept_id')['name_id'].apply(set).to_dict()
        self.blacklist_map = {cid: np.array(list(blacklist_grouped.get(cid, set()))) for cid in self.target_concept_ids}
        
    
    def __iter__(self):
        self.rng = np.random.default_rng(self.seed)
        for concept_id in self.target_concept_ids:
            concept_name = self.concept_name_lookup[concept_id]
            blacklist = self.blacklist_map.get(concept_id, set())
            
            n_select = min(self.max_element, self.num_candidates - len(blacklist), (self.num_candidates - len(blacklist))*3)
            
            chosen_name_ids = self.rng.choice(self.all_name_ids, size=n_select, replace=False)
            
            chosen_name_ids = np.setdiff1d(chosen_name_ids, blacklist, assume_unique=True)
            
            if len(chosen_name_ids) < n_select:
                filtered_candidates = np.setdiff1d(np.setdiff1d(self.all_name_ids, chosen_name_ids, assume_unique=True), blacklist, assume_unique=True)
                
                new_chosen = self.rng.choice(filtered_candidates, size=n_select-len(chosen_name_ids), replace=False)
                
                chosen_name_ids = np.concatenate([chosen_name_ids, new_chosen])
            
            chosen_name_ids = chosen_name_ids[:n_select]
            
            for name_id in chosen_name_ids:
                name = self.name_lookup[name_id]
                
                yield {
                    'sentence1': concept_name,
                    'sentence2': name,
                    'concept_id1': concept_id,
                    'concept_id2': name_id,
                    'label': 0
                }


class CombinedIterable:
    def __init__(self, 
                 target_concepts,
                 name_table,
                 positive_name_bridge = None,
                 false_positive_name_bridge = None,
                 blacklist_name_bridge = None,
                 positive_max_element = 999999,
                 false_positive_max_element = 999999,
                 negative_max_element = 5,
                 seed = 42
                 ):
        iterators = {}
        max_item = {}
        target_concepts = target_concepts.reset_index(drop=True)
        if positive_name_bridge is not None:
            it = PositiveIterable(
                target_concepts=target_concepts,
                name_table=name_table,
                name_bridge=positive_name_bridge,
                max_element=positive_max_element
            )
            iterators = iterators | {"positive": it}
            max_item['positive'] = positive_max_element
            
        if false_positive_name_bridge is not None:
            it = FalsePositiveIterable(
                target_concepts=target_concepts,
                name_table=name_table,
                name_bridge=false_positive_name_bridge,
                max_element=false_positive_max_element
            )
            iterators = iterators | {"false_positive": it}
            max_item['false_positive'] = false_positive_max_element
            
        if blacklist_name_bridge is not None:
            it = NegativeIterable(
                target_concepts=target_concepts,
                name_table=name_table,
                blacklist_name_bridge=blacklist_name_bridge,
                max_element=negative_max_element,
                seed=seed
            )
            iterators = iterators | {"negative": it}
            max_item['negative'] = negative_max_element
            
        self.iterators = iterators
        self.max_item = max_item
        self.target_rank = {key: val for key, val in zip(target_concepts.concept_id, target_concepts.index)}
    
    
    def _check_next_rank(self, iter_name, it):
        if iter_name not in self.cache:
            try:
                item = next(it)
                concept_id = item['concept_id1']
                self.cache[iter_name] = item
                return self.target_rank[concept_id]
            except StopIteration:
                return float('inf')  # No more items in this iterator
        else:
            item = self.cache[iter_name]
            concept_id = item['concept_id1']
            return self.target_rank[concept_id]
    
    def _get_next_item(self, iter_name, it):
        if iter_name not in self.cache:
            item = next(it)
        else:
            item = self.cache[iter_name]
            del self.cache[iter_name]
        return item
    
    def __iter__(self):
        self.cache = {}
        iters = {i:iter(it) for i, it in self.iterators.items()}
        stopped = {i:False for i in self.iterators.keys()}
        
        while not all(stopped.values()):
            ## establish common ranks
            ranks = {i: self._check_next_rank(i, v) for i, v in iters.items() if not stopped[i]}
            ranks = {i: r for i, r in ranks.items() if r != float('inf')} 
            if not ranks: # no active iterators
                break
            
            current_rank = min(ranks.values())
            
            for i, v in iters.items():
                if stopped[i]:
                    continue
                
                yield_num = 0
                while self._check_next_rank(i, v) == current_rank and yield_num < self.max_item[i]:
                    try:
                        item = self._get_next_item(i, v)
                        item['iter_id'] = i
                        item['rank'] = current_rank
                        yield_num += 1
                        yield item
                    except StopIteration:
                        stopped[i] = True
                        continue
                
        
        
        
                 
                 
                 