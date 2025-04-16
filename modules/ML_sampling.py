from typing import Dict, Iterator, Union, Optional, Set
import pandas as pd
import random
from tqdm import tqdm
import torch
import numpy as np


def remove_reserved(row, reserved_ids, id_col, name_col):
    nonstd_ids = row[id_col]
    nonstd_names = row[name_col]
    filtered = [(cid, name) for cid, name in zip(nonstd_ids, nonstd_names) if cid not in reserved_ids]

    # Unzip the filtered list if it's not empty, else assign empty lists
    if filtered:
        filtered_ids, filtered_names = zip(*filtered)
    else:
        filtered_ids, filtered_names = [], []
    return filtered_ids, filtered_names


def get_sentence_name(domain_id, concept_name):
    sentence_name = domain_id + ': ' + concept_name
    return sentence_name


def get_filtered_concept_ancestor(concept_ancestor, target_ids):
    """
    Filter the concept_ancestor table to include only relevant concepts. Another concept can be from any domain.
    
    Args:
        concept_ancestor (pd.DataFrame): The concept_ancestor table.
        target_ids (list): List of target concept IDs.
    Returns:
        pd.DataFrame: Filtered concept_ancestor table.
    """
    concept_ancestor_filtered = concept_ancestor[
        (
            concept_ancestor['ancestor_concept_id'].isin(target_ids)|
            concept_ancestor['descendant_concept_id'].isin(target_ids)
        )&
        concept_ancestor['min_levels_of_separation']!=0
        ]
    return concept_ancestor_filtered



def create_relation_maps(concept_ancestor):
    """
    Create a mapping between standard concepts and their parents/children.

    Args:
        concept_ancestor (pd.DataFrame): The concept_ancestor table.
        
    Returns:
        pd.DataFrame: A DataFrame with the mapping between standard concepts and their parents/children
        columns: ['from_concept_id', 'to_concept_id', 'min_levels_of_separation', 'max_levels_of_separation', 'type'
    
    """
    
    ## Remove the concept that maps to itself
    ## For a given standard concept, create a mapping to its children
    concept_ancestor_map = concept_ancestor.rename(
            columns={
            'ancestor_concept_id': 'from_concept_id',
            'descendant_concept_id': 'to_concept_id'
    })
    concept_ancestor_map['type']='ancestor_to_descendant'

    ## For a given standard concept, create a mapping to its parents
    concept_offspring_map = concept_ancestor.rename(
            columns={
            'descendant_concept_id': 'from_concept_id',
            'ancestor_concept_id': 'to_concept_id'
    })
    concept_offspring_map['type']='descendant_to_ancestor'

    ## combine the two mappings
    relation_maps = pd.concat([concept_ancestor_map, concept_offspring_map]
        ).groupby(['from_concept_id']
        ).agg({
        'to_concept_id': list,
        'min_levels_of_separation': list,
        'max_levels_of_separation': list,
        'type': list
        }).reset_index()
    
    return relation_maps


def generate_matching_positive_samples(df):
    """
    Create a dataset that contains 1-1 mappings between standard concepts and all non-standard concepts.

    Args:
        df (pd.DataFrame): The input DataFrame with at least columns ['std_name', 'concept_id', 'all_nonstd_concept_id', 'all_nonstd_name'].

    Returns:
        pd.DataFrame: A processed dataset with exploded rows and additional metadata.
    """
    # Filter rows where 'all_nonstd_name' has non-empty lists
    df = df[df['all_nonstd_name'].str.len() > 0].reset_index()

    # Explode the 'all_nonstd_concept_id' and 'all_nonstd_name' columns
    exploded_df = df[['std_name', 'concept_id', 'all_nonstd_concept_id', 'all_nonstd_name', 'source']].explode(
        ['all_nonstd_concept_id', 'all_nonstd_name', 'source']
    )

    # Prepare the training pairs
    exploded_df['sentence1'] = exploded_df['std_name']
    exploded_df['sentence2'] = exploded_df['all_nonstd_name']
    exploded_df['concept_id1'] = exploded_df['concept_id']
    exploded_df['concept_id2'] = exploded_df['all_nonstd_concept_id']
    exploded_df['label'] = 1

    # Select and format the final columns
    column_keep = ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label', 'source']
    final_dataset = exploded_df[column_keep].drop_duplicates()
    final_dataset['concept_id1'] = final_dataset['concept_id1'].astype('Int64')
    final_dataset['concept_id2'] = final_dataset['concept_id2'].astype('Int64')

    return final_dataset


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
                'label': 1
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
                    'label': 0
                }
                
            # 3. Yield False Positives (if available and not already yielded for this cid1)
            if cid1 in self.fp_map and cid1 not in self.yielded_fp_ids:
                fp_records = self.fp_map[cid1]
                for fp_record in fp_records:
                    yield {
                        'sentence1': fp_record['sentence1'], # Use sentence1 from FP record
                        'sentence2': fp_record['sentence2'],
                        'concept_id1': cid1,                 # The matching concept_id1
                        'concept_id2': fp_record['concept_id2'],
                        'label': 0                          # False positives are labeled 0
                    }
                self.yielded_fp_ids.add(cid1) # Mark this concept_id1's FPs as yielded for this epoch

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
    
    

class MatchingIterableDataset(GenericIterableDataset):
    def __init__(self,
                 positive_df_matching,
                 candidate_df_matching,
                 candidate_fp_matching = None,
                 n_neg = 150,
                 n_fp = 50,
                 seed=42,
                 special_token_sentence1=False,
                 special_token_sentence2=False,
                 special_token_candidate=False):
        
        positive_df = positive_df_matching[["sentence1", "sentence2", "concept_id1", "concept_id2", "label"]].copy()
        
        # Create candidate_df
        candidate_df = candidate_df_matching[["concept_id", "concept_name"]].reset_index(drop=True)
        
        # Add special tokens if specified
        if special_token_sentence1:
            positive_df['sentence1'] = add_special_token(positive_df['sentence1'], "[MATCHING]")
        if special_token_sentence2:
            positive_df['sentence2'] = add_special_token(positive_df['sentence2'], "[MATCHING]")
        if special_token_candidate:
            candidate_df['concept_name'] = add_special_token(candidate_df['concept_name'], "[MATCHING]")
        
        # Create blacklist_map
        target_ids = positive_df["concept_id1"].unique()
        blacklist_df = pd.DataFrame({
            "concept_id1": target_ids,
            "concept_id2": target_ids
        })
        blacklist_map = get_blacklist_map(target_ids, candidate_df, blacklist_df)
        
        if candidate_fp_matching is not None and not candidate_fp_matching.empty:
            ## find the max within_group_index
            max_within_group_index = candidate_fp_matching['within_group_index'].max()
            if n_fp>max_within_group_index+1:
                print(f"Warning: n_fp ({n_fp}) is greater than the maximum within_group_index ({max_within_group_index+1}).")
            candidate_fp_matching = candidate_fp_matching[candidate_fp_matching['within_group_index'] <= n_fp-1].copy()
        
        super().__init__(
            positive_df=positive_df,
            candidate_df=candidate_df,
            blacklist_map=blacklist_map,
            candidate_fp=candidate_fp_matching,
            n_neg=n_neg,
            seed=seed
        )




def generate_relation_positive_samples(
    concept_ancestor_filtered: pd.DataFrame,
    concept: pd.DataFrame
):
    """
    Generate positive samples for ancestor-descendant relationships.
    
    Args:
        concept_ancestor_filtered (pd.DataFrame): Filtered concept_ancestor table that you want to use to generate positive samples. Requires columns: ancestor_concept_id, descendant_concept_id.
        concept (pd.DataFrame): Concept table to provide concept names. Requires columns: concept_id, concept_name, domain_id.
    Returns:
        pd.DataFrame: Positive samples dataset with at least
        ["sentence1", "sentence2","label"].
    """
    positive_dataset = concept_ancestor_filtered.copy()
    concept_name_df = concept[['concept_id', 'concept_name', 'domain_id']]

    ## attach the ancestor and descendant concept names
    positive_dataset = positive_dataset.merge(
        concept_name_df, 
        left_on="ancestor_concept_id", 
        right_on="concept_id", 
        how="left"
    ).rename(columns={
        "concept_name": "ancestor_concept_name",
        "domain_id": "ancestor_domain_id"
        }).drop(
            columns=["concept_id"]
        )

    positive_dataset = positive_dataset.merge(
        concept_name_df, 
        left_on="descendant_concept_id", 
        right_on="concept_id", 
        how="left"
    ).rename(columns={
        "concept_name": "descendant_concept_name",
        "domain_id": "descendant_domain_id"
        }).drop(
            columns=["concept_id"]
        )

    positive_dataset['sentence1'] = get_sentence_name(positive_dataset['ancestor_domain_id'], positive_dataset['ancestor_concept_name'])

    positive_dataset['sentence2'] = get_sentence_name(positive_dataset['descendant_domain_id'], positive_dataset['descendant_concept_name'])
    
    positive_dataset['concept_id1'] = positive_dataset['ancestor_concept_id']
    positive_dataset['concept_id2'] = positive_dataset['descendant_concept_id']

    positive_dataset['label'] = 1

    len(positive_dataset) # 2819798
    
    return positive_dataset



class OffspringIterableDataset(GenericIterableDataset):
    def __init__(self,
                 positive_dataset_relation,
                 std_target,
                 n_neg=40,
                 seed=42):
        
        target_standard_ids = std_target['concept_id']
        candidate_df = std_target.copy()[['concept_id', 'std_name']].rename(columns={'std_name': 'concept_name'})
        positive_df = positive_dataset_relation[['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']].copy()

        blacklist_map = get_blacklist_map(target_standard_ids, candidate_df, positive_df)

        to_offspring_token = '[OFFSPRINT]'
        positive_df['sentence1'] = positive_df['sentence1'].apply(lambda x: add_special_token(x, to_offspring_token))

        super().__init__(
            positive_df=positive_df,
            candidate_df=candidate_df,
            blacklist_map=blacklist_map,
            n_neg=n_neg,
            seed=seed
        )


class AncestorIterableDataset(GenericIterableDataset):
    def __init__(self,
                 positive_dataset_relation,
                 std_target,
                 n_neg=40,
                 seed=42):
        
        target_standard_ids = std_target['concept_id']
        candidate_df = std_target.copy()[['concept_id', 'std_name']].rename(columns={'std_name': 'concept_name'})
        positive_df = positive_dataset_relation[['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']].copy()

        blacklist_df = positive_df.rename(columns={'concept_id1': 'concept_id2', 'concept_id2': 'concept_id1'})
        blacklist_map = get_blacklist_map(target_standard_ids, candidate_df, blacklist_df)

        to_ancestor_token = '[ANCESTOR]'
        positive_df['sentence2'] = positive_df['sentence2'].apply(lambda x: add_special_token(x, to_ancestor_token))
        candidate_df['concept_name'] = candidate_df['concept_name'].apply(lambda x: add_special_token(x, to_ancestor_token))

        super().__init__(
            positive_df=positive_df,
            candidate_df=candidate_df,
            blacklist_map=blacklist_map,
            n_neg=n_neg,
            seed=seed
        )





def add_special_token(x, token):
    x = token + " " + x
    return x


# get_blacklist_map(target_standard_ids, candidate_df, positive_df)
# positive_ids = target_standard_ids
# blacklist_df = positive_df
def get_blacklist_map(positive_ids, candidate_df, blacklist_df):
    """
    Create a map of forbidden concept pairs. The map is from concept_id1 to a set of row indices in candidate_df that are blacklisted.
    
    Args:
        positive_ids (list): List of positive concept IDs.
        candidate_df (pd.DataFrame): DataFrame containing candidate concepts. Required columns: 'concept_id'.
        blacklist_df (pd.DataFrame): DataFrame containing blacklisted concept pairs. Required columns: 'concept_id1', 'concept_id2'.
    """
    candidateIndex_df = candidate_df[['concept_id']].copy()
    candidateIndex_df['row_index'] = candidateIndex_df.index

    blacklist_df = blacklist_df[['concept_id1', 'concept_id2']].copy()
    blacklist_df_self = pd.DataFrame({
        "concept_id1": positive_ids,
        "concept_id2": positive_ids
    })
    blacklist_df = pd.concat([blacklist_df, blacklist_df_self])

    blacklist_df = blacklist_df[['concept_id1', 'concept_id2']].drop_duplicates()
    blacklist_df = blacklist_df.merge(candidateIndex_df, left_on='concept_id2', right_on='concept_id', how='inner')
    blacklist_df = blacklist_df[['concept_id1', 'row_index']]

    ## Create a map of concept_id1 to a set of blacklisted row_index
    blacklist_map = blacklist_df.groupby("concept_id1")["row_index"].apply(set).to_dict()
    return blacklist_map



## Combine pd.Series of list or string into a single list
def combine_elements(x):
        res = []
        for i in x:
            if isinstance(i, list) or isinstance(i, np.ndarray):
                res.extend([j for j in i if j is not None])
            elif isinstance(i, str):
                res.append(i)
        return res
    
def combine_columns(df):
    res = df.apply(combine_elements, axis=1)
    return res


def replace_std_concept_with_reserved(concept_ancestor_validation, std_bridge, column):
    """
    Return an ancestor table with at least one of the concept_id being the reserved concept
    """
    concept_ancestor_validation = concept_ancestor_validation.merge(
        std_bridge,
        left_on=column,
        right_on='concept_id2',
        how='left'
    )

    ## combine concept_id1 and ancestor_concept_id
    ## if concept_id1 is not none, use it, otherwise use ancestor_concept_id
    replacement = concept_ancestor_validation['concept_id1']
    concept_ancestor_validation.loc[~replacement.isna(), column] = replacement[~replacement.isna()]
    concept_ancestor_validation = concept_ancestor_validation[['ancestor_concept_id', 'descendant_concept_id']]
    
    return concept_ancestor_validation


