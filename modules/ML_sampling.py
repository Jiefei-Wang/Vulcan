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
    Filter the concept_ancestor table to include only relevant concepts.
    
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
    return final_dataset



class GenericIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                positive_df,
                candidate_df,
                blacklist_map,
                n_neg=4,
                seed=42):
        """
        Args:
            positive_df (pd.DataFrame): DataFrame containing positive samples. Requires columns: sentence1, sentence2, concept_id1, concept_id2.
            candidate_df (pd.DataFrame): DataFrame containing candidate samples. Requires columns: concept_id, concept_name.
            blacklist_map (dict): Dictionary mapping concept_id1 to a set of row indices in candidate_df that are blacklisted.
            n_neg (int): Number of negative samples to generate for each positive sample.
            seed (int): Random seed for reproducibility.
        """
        self.positive_df = positive_df
        self.blacklist_map = blacklist_map
        self.candidate_df = candidate_df.reset_index(drop=True)
        self.n_neg = n_neg
        self.seed = seed
        self.random = random.Random(seed)
    
    def _generate_negative_samples(self, row):
        """Generate `n_neg` negative samples dynamically for a given row"""
        concept_id1 = row["concept_id1"]
        blacklist_set = self.blacklist_map[concept_id1]

        candidate_row_ids = range(len(self.candidate_df))
        # Randomly sample 2 * n_neg candidates
        choices = self.random.sample(candidate_row_ids, 2 * self.n_neg)

        # Filter out invalid choices (same ID and blacklisted pairs)
        choices_filtered = [
            neg_id for neg_id in choices
            if neg_id not in blacklist_set
        ]

        # Ensure exactly `n_neg` negative samples
        if len(choices_filtered) < self.n_neg:
            extra_needed = self.n_neg - len(choices_filtered)
            remaining_candidates = list(set(candidate_row_ids) - set(choices_filtered) - blacklist_set)
            additional_choices = self.random.sample(remaining_candidates, min(extra_needed, len(remaining_candidates)))
            choices_filtered += additional_choices
        elif len(choices_filtered) > self.n_neg:
            choices_filtered = choices_filtered[:self.n_neg]

        # Generate negative samples
        for row_idx in choices_filtered:
            concept_info = self.candidate_df.iloc[row_idx]
            yield {
                "sentence1": row["sentence1"],
                "sentence2": concept_info["concept_name"],
                "concept_id1": concept_id1,
                "concept_id2": concept_info["concept_id"],
                "label": 0
            }

    def __iter__(self):
        for index, row in self.positive_df.iterrows():
            self.index = index
            yield row.to_dict()
            yield from self._generate_negative_samples(row)
            
    ## print function
    def __str__(self):
        return f"GenericIterableDataset: {self.index}/{len(self.positive_df)}"


class MatchingIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                positive_df_matching,
                matching_candidate_df,
                n_neg=40,
                seed=42,
                special_token_sentence1 = False,
                special_token_sentence2 = True,
                special_token_candidate = True):
        
        positive_df = positive_df_matching[["sentence1", "sentence2", "concept_id1", "concept_id2", "label"]].copy()
        
        ## Create candidate_df
        candidate_df = matching_candidate_df[["concept_id", "concept_name"]].reset_index(drop=True)
        
        ## add special token
        if special_token_sentence1:
            positive_df['sentence1'] = add_special_token(positive_df['sentence1'], "[MATCHING]")
        if special_token_sentence2:
            positive_df['sentence2'] = add_special_token(positive_df['sentence2'], "[MATCHING]")
        if special_token_candidate:
            candidate_df['concept_name'] = add_special_token(candidate_df['concept_name'], "[MATCHING]")
        
        ## Create blacklist_map
        target_ids = positive_df["concept_id1"].unique()
        blacklist_df = pd.DataFrame({
            "concept_id1": target_ids,
            "concept_id2": target_ids
        })
        blacklist_map = get_blacklist_map(target_ids, candidate_df, blacklist_df)
        
        
        self.iterator = GenericIterableDataset(
            positive_df,
            candidate_df,
            blacklist_map,
            n_neg=n_neg,
            seed=seed
        )
    
    def __iter__(self):
        return self.iterator.__iter__()
    
    def __str__(self):
        return self.iterator.__str__()




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




class OffspringIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                positive_dataset_relation,
                std_target,
                n_neg=40,
                seed=42):
        target_standard_ids = std_target['concept_id']
        
        candidate_df = std_target.copy()[['concept_id', 'std_name']].rename(columns={'std_name': 'concept_name'})
        
        positive_df = positive_dataset_relation[['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']].copy()

        blacklist_map = get_blacklist_map(target_standard_ids, candidate_df, positive_df)

        ## For [OFFSPRINT] token
        to_offspring_token = '[OFFSPRINT]'
        positive_df['sentence1'] = positive_df['sentence1'].apply(lambda x: add_special_token(x, to_offspring_token))
        
        
        self.iterator = GenericIterableDataset(
            positive_df,
            candidate_df,
            blacklist_map,
            n_neg=n_neg,
            seed=seed
        )
    
    def __iter__(self):
        return self.iterator.__iter__()
    
    def __str__(self):
        return self.iterator.__str__()




class AncestorIterableDataset(torch.utils.data.IterableDataset):
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



        ## For [ANCESTOR] token
        to_ancestor_token = '[ANCESTOR]'
        positive_df['sentence2'] = positive_df['sentence2'].apply(lambda x: add_special_token(x, to_ancestor_token))
        candidate_df['concept_name'] = candidate_df['concept_name'].apply(lambda x: add_special_token(x, to_ancestor_token))

        
        
        self.iterator = GenericIterableDataset(
            positive_df,
            candidate_df,
            blacklist_map,
            n_neg=n_neg,
            seed=seed
        )
    
    def __iter__(self):
        return self.iterator.__iter__()
    
    def __str__(self):
        return self.iterator.__str__()






# def generate_relation_negative_samples(
#     positive_dataset_relation: pd.DataFrame,
#     concept_ancestor_filtered: pd.DataFrame,
#     concept: pd.DataFrame,
#     n_neg: int = 4,
#     seed: int = 42
# ) -> pd.DataFrame:
#     """
#     Generate negative samples for ancestor-descendant relationships.
    
#     Args:
#         positive_dataset_relation (pd.DataFrame): Positive samples dataset.
#         concept_ancestor_filtered (pd.DataFrame): Filtered concept_ancestor table.
#         concept (pd.DataFrame): Concept table to provide concept names.
#         n_neg (int): Number of negative samples to generate per positive sample row.
#         seed (int): Random seed for reproducible sampling.
#     """
#     random.seed(seed)
#     ## all candidate concepts
#     candidate_ids = set(concept_ancestor_filtered['descendant_concept_id'].unique()) | set(concept_ancestor_filtered['ancestor_concept_id'].unique())
#     candidate_ids = list(candidate_ids)

#     ## duplicate negative_dataset_relation n_reg + 2 times
#     negative_dataset_candidate = positive_dataset_relation[['sentence1', 'concept_id1']].copy()
#     ## randomly sample negative samples from candidate_ids
#     negative_dataset_candidate['concept_id2'] = negative_dataset_candidate['concept_id1'].apply(lambda x: random.sample(candidate_ids, n_neg + 2))

#     negative_dataset_candidate = negative_dataset_candidate.explode('concept_id2')

#     ## remove those whose concept_id1 = concept_id2
#     negative_dataset_candidate = negative_dataset_candidate[negative_dataset_candidate['concept_id1'] != negative_dataset_candidate['concept_id2']]

#     ## create a black list of parent/child relationship
#     black_list = concept_ancestor_filtered[["ancestor_concept_id", "descendant_concept_id"]].copy()
#     black_list['hit'] = 1

#     negative_dataset_candidate = negative_dataset_candidate.merge(
#         black_list,
#         left_on=['concept_id1', 'concept_id2'],
#         right_on=['ancestor_concept_id', 'descendant_concept_id'],
#         how='left'
#     ).merge(
#         black_list,
#         left_on=['concept_id2', 'concept_id1'],
#         right_on=['ancestor_concept_id', 'descendant_concept_id'],
#         how='left'
#     )

#     ## remove those in the black list
#     negative_dataset_candidate = negative_dataset_candidate[negative_dataset_candidate['hit_x'].isna()&negative_dataset_candidate['hit_y'].isna()]

#     negative_dataset_relation = negative_dataset_candidate[['sentence1', 'concept_id1', 'concept_id2']][:len(positive_dataset_relation) * n_neg]

#     ## add sentence2 and label
#     negative_dataset_relation = negative_dataset_relation.merge(
#         concept[['concept_id', 'concept_name', 'domain_id']],
#         left_on='concept_id2',
#         right_on='concept_id',
#         how='left'
#     )

#     negative_dataset_relation['sentence2'] = get_sentence_name(negative_dataset_relation['domain_id'], negative_dataset_relation['concept_name'])

#     negative_dataset_relation['label'] = 0

#     negative_dataset_relation = negative_dataset_relation[['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label']]
    
#     return negative_dataset_relation









class RelationNegativeIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, 
                 positive_dataset_relation, 
                 concept_ancestor_filtered, 
                 concept, 
                 n_neg=4, 
                 seed=42):
        """
        Iterable dataset for generating negative samples for ancestor-descendant relationships.

        Args:
            positive_dataset_relation (Iterable): Iterable dataset of positive relation pairs.
            concept_ancestor_filtered (pd.DataFrame): Filtered concept_ancestor table.
            concept (pd.DataFrame): Concept table to provide concept names.
            n_neg (int): Number of negative samples per positive sample.
            seed (int): Random seed.
        """
        print("Initializing RelationNegativeIterableDataset")
        random.seed(seed)

        # Ensure positive dataset is iterable
        if isinstance(positive_dataset_relation, pd.DataFrame):
            self.positive_dataset_relation = positive_dataset_relation.to_dict(orient="records")
        else:
            self.positive_dataset_relation = positive_dataset_relation

        self.concept_ancestor_filtered = concept_ancestor_filtered
        self.concept = concept
        self.n_neg = n_neg

        # Special token for relation dataset
        self.relation_token = "[RELATION]"

        # Get all candidate concept IDs
        candidate_ids = set(concept_ancestor_filtered['descendant_concept_id'].unique()) | \
                        set(concept_ancestor_filtered['ancestor_concept_id'].unique())
        self.candidate_ids = list(candidate_ids)

        # Create a blacklist of parent-child relationships for fast lookup
        black_list = concept_ancestor_filtered[["ancestor_concept_id", "descendant_concept_id"]]
        self.blacklist_set = set(zip(black_list["ancestor_concept_id"], black_list["descendant_concept_id"]))

        # Create a fast lookup table for concept names
        self.concept_dict = concept.set_index("concept_id")[["concept_name", "domain_id"]].to_dict(orient="index")

    def _generate_negative_samples(self, row):
        """Generate `n_neg` negative samples dynamically for a given row"""
        concept_id1 = row["concept_id1"]

        # Randomly sample 2 * n_neg candidates
        choices = random.sample(self.candidate_ids, 2 * self.n_neg)

        # Filter out invalid choices (same ID and blacklisted pairs)
        choices_filtered = [
            neg_id for neg_id in choices
            if neg_id != concept_id1 and (concept_id1, neg_id) not in self.blacklist_set and (neg_id, concept_id1) not in self.blacklist_set
        ]

        # Ensure exactly `n_neg` negative samples
        if len(choices_filtered) < self.n_neg:
            extra_needed = self.n_neg - len(choices_filtered)
            remaining_candidates = list(set(self.candidate_ids) - set(choices_filtered) - {concept_id1})
            additional_choices = random.sample(remaining_candidates, min(extra_needed, len(remaining_candidates)))
            choices_filtered += additional_choices
        elif len(choices_filtered) > self.n_neg:
            choices_filtered = choices_filtered[:self.n_neg]

        # Generate negative samples
        for neg_id in choices_filtered:
            if neg_id in self.concept_dict:
                concept_info = self.concept_dict[neg_id]
                mapped_name = get_sentence_name(concept_info["domain_id"], concept_info["concept_name"])
                yield {
                    "sentence1": self.relation_token + " " + row["sentence1"],
                    "sentence2": self.relation_token + " " + mapped_name,
                    "concept_id1": concept_id1,
                    "concept_id2": int(neg_id),
                    "label": 0,
                    "source": "relation_negative"
                }

    def __iter__(self):
        """Yields only negative samples dynamically"""
        for row in self.positive_dataset_relation:
            yield from self._generate_negative_samples(row)


def add_special_token(x, token):
    x = token + " " + x
    return x


def add_special_token_df(df, token):
    df['sentence1'] = add_special_token(df['sentence1'], token)
    df['sentence2'] = add_special_token(df['sentence2'], token)
    return df


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
    blacklist_df = blacklist_df.merge(candidateIndex_df, left_on='concept_id2', right_on='concept_id', how='left')
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
