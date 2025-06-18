import numpy as np

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
            name_table (pd.DataFrame): DataFrame containing name mappings with columns ['name_id', 'source', 'source_id', 'type', 'name']
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
        self.name_lookup = dict(zip(name_table['name_id'], 
                                   zip(name_table['name'], name_table['source_id'])))
        
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
                    
                name, source_id = self.name_lookup[name_id]
                
                yield {
                    'sentence1': concept_name,
                    'sentence2': name,
                    'concept_id1': concept_id,
                    'concept_id2': source_id,
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
            name_table (pd.DataFrame): DataFrame containing name mappings with columns ['name_id', 'source', 'source_id', 'type', 'name']
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
        self.name_lookup = dict(zip(name_table['name_id'], 
                                   zip(name_table['name'], name_table['source_id'])))
        
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
                name, source_id = self.name_lookup[name_id]
                
                yield {
                    'sentence1': concept_name,
                    'sentence2': name,
                    'concept_id1': concept_id,
                    'concept_id2': source_id,
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
        iterators = []
        target_concepts = target_concepts.reset_index(drop=True)
        if positive_name_bridge is not None:
            iterators.append(PositiveIterable(
                target_concepts=target_concepts,
                name_table=name_table,
                name_bridge=positive_name_bridge,
                max_element=positive_max_element
            ))
            
        if false_positive_name_bridge is not None:
            iterators.append(FalsePositiveIterable(
                target_concepts=target_concepts,
                name_table=name_table,
                name_bridge=false_positive_name_bridge,
                max_element=false_positive_max_element
            ))
        if blacklist_name_bridge is not None:
            iterators.append(NegativeIterable(
                target_concepts=target_concepts,
                name_table=name_table,
                blacklist_name_bridge=blacklist_name_bridge,
                max_element=negative_max_element,
                seed=seed
            ))
        self.iterators = iterators
        self.target_rank = {key: val for key, val in zip(target_concepts.concept_id, target_concepts.index)}
    
    def __iter__(self):
        iters = [iter(it) for it in self.iterators]
        iter_ranks = [0] * len(iters)
        current_min_rank = 0
        stored_values = {}
        stopped = [False] * len(iters)
        
        while not all(stopped):
            # Update current minimum rank from active iterators
            active_ranks = [iter_ranks[i] for i in range(len(iter_ranks)) if not stopped[i] or i in stored_values]
            if active_ranks:
                current_min_rank = min(active_ranks)
            
            for i, it in enumerate(iters):
                if stopped[i]:
                    continue
                    
                # Check if we have a stored value for this iterator
                if i in stored_values:
                    item, item_rank = stored_values[i]
                    if item_rank == current_min_rank:
                        # Yield stored item if it matches current minimum rank
                        yield item
                        del stored_values[i]
                        # Don't try to get new item from this iterator this round
                        continue
                    else:
                        # Skip this iterator if stored value doesn't match current rank
                        continue
                        
                try:
                    item = next(it)
                    concept_id1 = item['concept_id1']
                    item_rank = self.target_rank[concept_id1]
                    iter_ranks[i] = item_rank
                    item['iter_id'] = i  # Store iterator ID for reference 
                    item['rank'] = item_rank  # Store rank for sorting later
                    if item_rank == current_min_rank:
                        # Yield immediately if matches current rank
                        yield item
                    else:
                        # Store the item for later if it doesn't match current rank
                        stored_values[i] = (item, item_rank)
                        
                except StopIteration:
                    stopped[i] = True
                    continue  
            
            # If no progress is made, increment the minimum rank
            if all(i in stored_values or stopped[i] for i in range(len(iters))):
                if stored_values:
                    current_min_rank = min(rank for _, rank in stored_values.values())
        
        # After all iterators are stopped, yield any remaining stored values
        # Sort by rank to maintain order
        remaining_items = [(item, rank) for item, rank in stored_values.values()]
        remaining_items.sort(key=lambda x: x[1])  # Sort by rank
        for item, _ in remaining_items:
            yield item
        
                 
                 
                 