from pandas import pd

class GenericIterableDataset():
    def __init__(self,
                 positive_df: pd.DataFrame,
                 candidate_df: pd.DataFrame,
                 blacklist_map: Dict[int, Set[int]],
                 false_positive_df: Optional[pd.DataFrame] = None,
                 n_neg: int = 4,
                 seed: int = 42):
        """
        Initialize the dataset with required DataFrames and parameters.

        Args:
            positive_df: DataFrame with positive examples.
                         Required columns: ['sentence1', 'sentence2','concept_id1', 'concept_id2']
            candidate_df: DataFrame with candidate concepts for negative sampling. Required columns: ['concept_id', 'concept_name']
            blacklist_map: Dictionary mapping concept_id1 to a set of candidate indices to exclude during negative sampling for that concept_id1.
            false_positive_df: Optional DataFrame with pre-defined false positive examples. Required columns if provided: ['concept_id1', 'sentence1', 'concept_id2', 'sentence2']
            n_neg: Number of random negative samples to generate per positive example.
            seed: Random seed for reproducibility.
        """
        ## validate input
        self.validate_data(positive_df, candidate_df, false_positive_df)
        
        """Initialize the dataset with required DataFrames and parameters."""
        self.n_neg = n_neg
        self.rng = np.random.default_rng(seed)

        # Store positive data as numpy arrays for faster indexing
        self.pos_sentences1 = positive_df['sentence1'].values
        self.pos_sentences2 = positive_df['sentence2'].values
        self.pos_concept_id1 = positive_df['concept_id1'].values
        self.pos_concept_id2 = positive_df['concept_id2'].values
        self.positive_df_len = len(positive_df) # Store original length for __str__

        # Candidate data as numpy arrays for efficient sampling
        self.n_candidates = len(candidate_df)
        self.candidate_index = candidate_df.reset_index(drop=True).index.values
        self.candidate_concept_ids = candidate_df['concept_id'].values
        self.candidate_concept_names = candidate_df['concept_name'].values

        # Convert blacklist_map to use numpy arrays for efficiency
        ## this is a map from concept id to a set of indices in candidate_df
        self.blacklist_map = {cid: np.array(list(blacklist_map[cid]), dtype=int) for cid in blacklist_map}

        self.num_candidates = len(self.candidate_concept_ids)
        
        # Process false positive data
        self.fp_map = {}
        self.total_fp_count = 0
        if false_positive_df is not None and not false_positive_df.empty:
            # Group FP samples by concept_id1 and store relevant data
            grouped_fp = false_positive_df.groupby('concept_id1')
            for cid, group in grouped_fp:
                # Store as list of dicts for easy iteration
                fp_records = group[['sentence1', 'sentence2', 'concept_id2']].to_dict('records')
                self.fp_map[cid] = fp_records
                self.total_fp_count += len(fp_records) # Count total FPs for length calculation

        self.index = 0
        self.yielded_fp_ids = set()
    
    def validate_data(self, positive_df, candidate_df, false_positive_df):
        """
        validate if all requirement are met
        """
        ## check column names
        required_cols = {
            'positive_df': ['sentence1', 'sentence2', 'concept_id1', 'concept_id2'],
            'candidate_df': ['concept_id', 'concept_name']
            }
        for df_name, cols in required_cols.items():
            df = locals()[df_name]
            missing = [col for col in cols if col not in df.columns]
            if missing:
                raise ValueError(f"{df_name} missing columns: {missing}")
        
        ## check column types
        for col in ['concept_id1', 'concept_id2']:
            if not pd.api.types.is_integer_dtype(positive_df[col]):
                raise TypeError(f"positive_df['{col}'] must be integer type")
        if not pd.api.types.is_integer_dtype(candidate_df['concept_id']):
            raise TypeError("candidate_df['concept_id'] must be integer type")
        
        if false_positive_df is not None and not false_positive_df.empty:
            for col in ['concept_id1', 'concept_id2']:
                if not pd.api.types.is_integer_dtype(false_positive_df[col]):
                    raise TypeError(f"false_positive_df['{col}'] must be integer type")
    
    
    def __iter__(self) -> Iterator[Dict[str, Union[str, int]]]:
        self.index = 0
        self.yielded_fp_ids.clear()
        
        for idx in range(len(self.pos_concept_id1)):
            self.index = self.index + 1
            
            # 1. yield positive example
            yield {
                'sentence1': self.pos_sentences1[idx],
                'sentence2': self.pos_sentences2[idx],
                'concept_id1': self.pos_concept_id1[idx],
                'concept_id2': self.pos_concept_id2[idx],
                'label': 1,
                'from': "positive"
            }

            # 2. yield negative sampling
            cid1 = self.pos_concept_id1[idx]
            blacklist = self.blacklist_map.get(cid1, np.array([], dtype=int))
            n_select = min(self.n_neg, self.num_candidates - len(blacklist))
            
            chosen_indices = self.rng.choice(self.candidate_index, size=n_select*3, replace=False)
            
            chosen_indices = np.setdiff1d(chosen_indices, blacklist, assume_unique=True)
            
            if len(chosen_indices) < n_select:
                filtered_candidates = np.setdiff1d(np.setdiff1d(self.candidate_index, chosen_indices, assume_unique=True), blacklist, assume_unique=True)
                
                new_chosen = self.rng.choice(filtered_candidates, size=n_select-len(chosen_indices), replace=False)
                
                chosen_indices = np.concatenate([chosen_indices, new_chosen])
            
            chosen_indices = chosen_indices[:n_select]
            
            for c_idx in chosen_indices:
                yield {
                    'sentence1': self.pos_sentences1[idx],
                    'sentence2': self.candidate_concept_names[c_idx],
                    'concept_id1': cid1,
                    'concept_id2': self.candidate_concept_ids[c_idx],
                    'label': 0,
                    "from": "negative"
                }
                
            # 3. Yield False Positives (if available and not already yielded for this cid1)
            if cid1 in self.fp_map and cid1 not in self.yielded_fp_ids:
                fp_records = self.fp_map[cid1]
                for fp_record in fp_records:
                    yield {
                        'sentence1': fp_record['sentence1'],
                        'sentence2': fp_record['sentence2'],
                        'concept_id1': cid1,                 
                        'concept_id2': fp_record['concept_id2'],
                        'label': 0,
                        'from': "false_positive"                      
                    }
                self.yielded_fp_ids.add(cid1) # Mark this cid1 as processed to avoid duplicates in future iterations

    def _calculate_length(self):
        """Calculates the exact total number of items the iterator will yield."""
        total_len = 0
        # Add count for positive examples
        total_len += self.positive_df_len

        # Add count for random negative samples
        neg_count = 0
        for cid1 in self.pos_concept_id1:
            blacklist = self.blacklist_map.get(cid1, np.array([], dtype=int))
            valid_count = self.num_candidates - len(blacklist)
            neg_count += min(self.n_neg, max(0, valid_count))
        total_len += neg_count

        # Add count for unique false positive samples
        # Iterate through unique concept_id1s present in positive_df that also have FPs
        processed_fp_for_len = set()
        fp_yield_count = 0
        for cid1 in self.pos_concept_id1:
             # Check if this cid1 has FPs and hasn't been counted yet
            if cid1 in self.fp_map and cid1 not in processed_fp_for_len:
                fp_yield_count += len(self.fp_map[cid1])
                processed_fp_for_len.add(cid1) # Mark as counted
        total_len += fp_yield_count

        self._len = total_len

    
    ## calculate the exact length of the dataset
    def _element_size(self):
        self._len = len(self.pos_sentences1)
        for cid1 in self.pos_concept_id1:
            blacklist = self.blacklist_map.get(cid1, np.array([], dtype=int))
            valid_count = self.num_candidates - len(blacklist)
            self._len += min(self.n_neg, max(0, valid_count))
    
    def __len__(self):
        """Returns the total number of items the iterator will yield in one epoch."""
        if not hasattr(self, '_len'):
            self._calculate_length()
        return self._len
    
    def element_size(self):
        return self.__len__()    
    
    def trainer_iter(self):
        it = self.__iter__()
        for i in it:
            yield {nm:i[nm] for nm in ['sentence1', 'sentence2', 'label']}
        
    ## print function
    def __str__(self):
        base_info = f"GenericIterableDataset - Positive Pairs: {self.positive_df_len}"
        len_info = f"Total Yield per Epoch: {len(self)}" if hasattr(self, '_len') else "Total Yield per Epoch: (call len() to calculate)"
        fp_info = f"Mapped FP Concepts: {len(self.fp_map)}, Total FP Rows: {self.total_fp_count}" if self.fp_map else "No False Positives Provided"
        # Show progress based on positive examples processed
        progress_info = f"Current Iteration Progress: {self.index}/{self.positive_df_len} (positive pairs)"
        return f"{base_info}\n{fp_info}\n{len_info}\n{progress_info}"
    
    