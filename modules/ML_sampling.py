from typing import Dict, Iterator, Union
import pandas as pd
import random
from tqdm import tqdm
import torch
import numpy as np


def remove_reserved(row, reserved_ids):
    nonstd_concept_id = row['nonstd_concept_id']
    nonstd_name = row['nonstd_name']
    if nonstd_concept_id is None:
        return None, None
    nonstd_name = [x for i, x in enumerate(nonstd_name) if nonstd_concept_id[i] not in reserved_ids]
    nonstd_concept_id = [x for x in nonstd_concept_id if x not in reserved_ids]
    return nonstd_concept_id, nonstd_name


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



# df = std_target_for_matching.copy()
# columns = ['nonstd_name', 'synonym_name', 'descriptions']
# column_ids = ['nonstd_concept_id', None, None]
def generate_matching_positive_samples(df, columns, column_ids):
    """
    Create a dataset that contains 1-1 mappings between standard concepts and non-standard concepts.

    Args:
        df (pd.DataFrame): The input DataFrame with at least columns ['std_name', 'concept_id']
        columns (list of str): The columns to process (e.g., ['nonstd_name', 'synonym_name', 'descriptions']).
        column_ids (list of str): The corresponding columns with concept IDs (e.g., ['nonstd_concept_id', None, None]).

    Returns:
        pd.DataFrame: A processed dataset with exploded rows and additional metadata.
    """
    df = df[df['all_nonstd'].str.len() > 0].reset_index()

    column_keep = ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label', 'source']
    result_frames = []
    for idx in range(len(columns)):
        column = columns[idx]
        column_id = column_ids[idx]
        ## filter out None values
        df2 = df[df[column].notna()]
        
        ## create 1-1 mapping between std and non-std
        columns_to_explode = [col for col in [column, column_id] if col is not None]
        exploded_df = df2[['std_name', 'concept_id'] + columns_to_explode].explode(columns_to_explode)
        
        ## prepare the training pair
        exploded_df['sentence1'] = exploded_df['std_name']
        exploded_df['sentence2'] = exploded_df[column]
        exploded_df['concept_id1'] = exploded_df['concept_id']
        if column_id is not None:
            exploded_df['concept_id2'] = exploded_df[column_id]
        else:
            exploded_df['concept_id2'] = None
        exploded_df['source'] = column
        exploded_df['label'] = 1
        exploded_df = exploded_df[column_keep]
        ## save as excel for inspection
        ## exploded_df.to_excel(f'positive_samples_{column}.xlsx', index=False)
        result_frames.append(exploded_df)
        
    final_dataset = pd.concat(result_frames, ignore_index=True).drop_duplicates()
    final_dataset['concept_id1'] = final_dataset['concept_id1'].astype('Int64')
    final_dataset['concept_id2'] = final_dataset['concept_id2'].astype('Int64')

    return final_dataset

# class GenericIterableDataset():
#     def __init__(self,
#                  positive_df,
#                  candidate_df,
#                  blacklist_map,
#                  n_neg=4,
#                  seed=42):
#         """Initialize the dataset with required DataFrames and parameters."""
#         self.positive_records = positive_df.to_dict("records")
#         self.candidate_df = candidate_df.reset_index(drop=True)
#         self.candidate_ids = self.candidate_df.index.values
#         self.blacklist_map = {k: set(v) for k, v in blacklist_map.items()}
#         self.n_neg = n_neg
#         self.rng = np.random.default_rng(seed)

#     def _generate_negative_samples(self, concept_id1, sentence1):
#         blacklist_set = self.blacklist_map.get(concept_id1, set())
#         valid_candidates = np.setdiff1d(self.candidate_ids, list(blacklist_set), assume_unique=True)

#         if len(valid_candidates) <= self.n_neg:
#             chosen = valid_candidates
#         else:
#             chosen = self.rng.choice(valid_candidates, size=self.n_neg, replace=False)

#         concepts = self.candidate_df.iloc[chosen]
#         return [{
#             "sentence1": sentence1,
#             "sentence2": row["concept_name"],
#             "concept_id1": concept_id1,
#             "concept_id2": row["concept_id"],
#             "label": 0
#         } for _, row in concepts.iterrows()]

#     def __iter__(self):
#         for row in self.positive_records:
#             yield {**row, "label": 1}
#             negatives = self._generate_negative_samples(row["concept_id1"], row["sentence1"])
#             yield from negatives

#     def __len__(self):
#         return len(self.positive_df) * (self.n_neg + 1)
    
#     def trainer_iter(self):
#         it = self.__iter__()
#         for i in it:
#             yield {nm:i[nm] for nm in ['sentence1', 'sentence2', 'label']}
        
#     ## print function
#     def __str__(self):
#         return f"GenericIterableDataset Total: {len(self)}\nProgress: {self.index}/{len(self.positive_df)}"
    
    
class GenericIterableDataset():
    def __init__(self,
                 positive_df,
                 candidate_df,
                 blacklist_map,
                 n_neg=4,
                 seed=42):
        ## validate input
        self.validate_data(positive_df, candidate_df)
        
        """Initialize the dataset with required DataFrames and parameters."""
        self.n_neg = n_neg
        self.rng = np.random.default_rng(seed)

        # Store positive data as numpy arrays for faster indexing
        self.pos_sentences1 = positive_df['sentence1'].values
        self.pos_sentences2 = positive_df['sentence2'].values
        self.pos_concept_id1 = positive_df['concept_id1'].values
        self.pos_concept_id2 = positive_df['concept_id2'].values

        # Candidate data as numpy arrays for efficient sampling
        self.n_candidates = len(candidate_df)
        self.candidate_index = candidate_df.reset_index(drop=True).index.values
        self.candidate_concept_ids = candidate_df['concept_id'].values
        self.candidate_concept_names = candidate_df['concept_name'].values

        # Convert blacklist_map to use numpy arrays for efficiency
        ## this is a map from concept id to a set of indices in candidate_df
        self.blacklist_map = {cid: np.array(list(blacklist_map[cid]), dtype=int) for cid in blacklist_map}

        self.num_candidates = len(self.candidate_concept_ids)
        
        self.index = 0
    
    def validate_data(self, positive_df, candidate_df):
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
    
    
    def __iter__(self) -> Iterator[Dict[str, Union[str, int]]]:
        self.index = 0
        for idx in range(len(self.pos_concept_id1)):
            self.index = self.index + 1
            # yield positive example
            yield {
                'sentence1': self.pos_sentences1[idx],
                'sentence2': self.pos_sentences2[idx],
                'concept_id1': self.pos_concept_id1[idx],
                'concept_id2': self.pos_concept_id2[idx],
                'label': 1
            }

            # negative sampling
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
    
    ## calculate the exact length of the dataset
    def _element_size(self):
        self._len = len(self.pos_sentences1)
        for cid1 in self.pos_concept_id1:
            blacklist = self.blacklist_map.get(cid1, np.array([], dtype=int))
            valid_count = self.num_candidates - len(blacklist)
            self._len += min(self.n_neg, max(0, valid_count))
    
    def element_size(self):
        if not hasattr(self, '_len'):
            self._element_size()
        return self._len
    
    
    def trainer_iter(self):
        it = self.__iter__()
        for i in it:
            yield {nm:i[nm] for nm in ['sentence1', 'sentence2', 'label']}
        
    ## print function
    def __str__(self):
        return f"GenericIterableDataset Total: {len(self)}\nProgress: {self.index}/{len(self.positive_df)}"
    
    

class MatchingIterableDataset(GenericIterableDataset):
    def __init__(self,
                 positive_df_matching,
                 matching_candidate_df,
                 n_neg=40,
                 seed=42,
                 special_token_sentence1=False,
                 special_token_sentence2=True,
                 special_token_candidate=True):
        
        positive_df = positive_df_matching[["sentence1", "sentence2", "concept_id1", "concept_id2", "label"]].copy()
        
        # Create candidate_df
        candidate_df = matching_candidate_df[["concept_id", "concept_name"]].reset_index(drop=True)
        
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
        
        super().__init__(
            positive_df=positive_df,
            candidate_df=candidate_df,
            blacklist_map=blacklist_map,
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


def add_special_token_df(df, token):
    df['sentence1'] = add_special_token(df['sentence1'], token)
    df['sentence2'] = add_special_token(df['sentence2'], token)
    return df

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


