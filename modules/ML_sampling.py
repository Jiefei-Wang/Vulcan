import pandas as pd
import random
from tqdm import tqdm
import torch


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



# df = std_condition_with_nonstd_updated.copy()
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


# positive_samples = positive_sample_dataset

# def generate_matching_negative_samples(
#     positive_dataset_matching: pd.DataFrame,
#     concept_ancestor: pd.DataFrame,
#     std_target_for_matching: pd.DataFrame,
#     n_neg: int = 4,
#     seed: int = 42
# ) -> pd.DataFrame:
#     """
#     Generate negative samples for each row in positive_samples.
    
#     Parameters
#     ----------
#     positive_samples : pd.DataFrame
#         Must contain at least columns: ["sentence1", "concept_id1"].
        
#     concept_ancestor : pd.DataFrame
        
#     std_target_for_matching : pd.DataFrame
#         Must contain ["concept_id", "all_nonstd", "std_name"] for standard concepts.
        
#     n_neg : int
#         Number of negative samples to generate per positive sample row.
        
#     seed : int
#         Random seed for reproducible sampling.
        
#     Returns
#     -------
#     pd.DataFrame
#         Columns: ["sentence1", "sentence2", "concept_id1", "concept_id2", "label", "source"] 
#         with label = 0 for negatives, source = "negative".
#     """
#     random.seed(seed)
    
#     target_ids_for_matching = std_target_for_matching['concept_id']
#     concept_ancestor_filtered = get_filtered_concept_ancestor(concept_ancestor, target_ids_for_matching)
#     relation_maps = create_relation_maps(concept_ancestor_filtered)
#     len(relation_maps) # 168816
    
#     # Make a dict to map std concept id to its exclusion list
#     relation_maps2 = relation_maps[['from_concept_id', 'to_concept_id']].copy()
#     relation_maps2.set_index("from_concept_id", inplace=True)
#     relation_maps2['to_concept_id'] = relation_maps2['to_concept_id'].apply(lambda x: set(x))
#     std_id_to_excluded = relation_maps2.to_dict()['to_concept_id']
    
#     ## prepare the mapping from concept_id to all possible names
#     std_maps_df = std_target_for_matching[['concept_id', "all_nonstd", "std_name"]].copy()
#     std_maps_df["all_names"] = std_maps_df.apply(
#         lambda x: x["all_nonstd"] + [x["std_name"]], axis=1
#     )
#     std_maps_df['maps_num'] = std_maps_df['all_names'].apply(lambda x: len(x))
    
#     ## Attach sample_idx to each name for sampling
#     ## Column: concept_id, all_names, sample_idx
#     id_to_names_df=std_maps_df[['concept_id', 'all_names']].copy()
#     id_to_names_df['sample_idx'] = id_to_names_df['all_names'].apply(lambda x: [i for i in range(len(x))])
#     id_to_names_df = id_to_names_df.explode(
#     ["all_names", 'sample_idx']
#     )
    
    
#     # for each row in positive_samples:
#     # 1. randomly sample 2*n_neg standard concept IDs
#     # 2. filter out the ones in the exclusion list
#     # 3. keep n_neg samples from the remaining set
#     all_std_ids = std_target_for_matching["concept_id"].to_list()
#     all_std_ids_set = set(all_std_ids)
    
#     # 1. randomly sample 2*n_neg standard concept IDs
#     negative_samples = positive_dataset_matching[['sentence1', 'concept_id1']].copy()
#     negative_samples['index'] = negative_samples.index
#     negative_samples["choices"] = [random.sample(all_std_ids, 2*n_neg) for _ in range(len(positive_dataset_matching))]
    
#     ## Refine the sampled choices such that they are not in the exclude set
#     for i, row in tqdm(negative_samples.iterrows(), total=len(negative_samples)):
#         std_id = row["concept_id1"]
#         choices = row["choices"]
#         exclude_ids = std_id_to_excluded.get(std_id, set())
#         choices_filtered = list(set(choices) - exclude_ids)
        
#         ## if the filtered set is too small, add more
#         ## if the filtered set is too large, keep the first n_neg
#         if len(choices_filtered) < n_neg:
#             n_neg_more = n_neg - len(choices_filtered)
#             candidate_ids = list(all_std_ids_set - exclude_ids - set(choices_filtered))
#             n_neg_more = min(n_neg_more, len(candidate_ids))
#             additional_ids = random.sample(candidate_ids, n_neg_more)
#             choices_filtered += additional_ids
#         elif len(choices_filtered) > n_neg:
#             choices_filtered = choices_filtered[:n_neg]
        
#         ## update the choices
#         negative_samples.at[i, "choices"] = choices_filtered
        
    
#     neg_df = negative_samples[['index','choices']]
#     neg_df = neg_df.explode("choices")
#     ## randomly pick number from 0 to 100000 with replacement
#     neg_df['sample_idx'] = [random.randint(0, 100000) for _ in range(len(neg_df))]
    
#     # Obtain the number of possible negative samples for each concept_id
#     neg_df = neg_df.merge(
#         std_maps_df[['concept_id', 'maps_num']],
#         left_on='choices',
#         right_on='concept_id',
#         how='left'
#     ).drop(columns='concept_id')
    
#     neg_df.columns
#     # ['index', 'choices', 'sample_idx', 'maps_num']
    
#     ## recalculate the sample_idx to be within the range of maps_num
#     neg_df['sample_idx'] = neg_df.apply(
#         lambda x: x['sample_idx'] % x['maps_num'], axis=1
#     )
#     len(neg_df) # 2825424
    
#     ## obtain the name of the mapped nonstd name
#     neg_df = neg_df.merge(
#         id_to_names_df,
#         left_on=['choices', 'sample_idx'],
#         right_on=['concept_id', 'sample_idx'],
#         how='inner'
#     ).drop(
#         columns=['maps_num', 'choices', 'sample_idx']
#     ).rename(
#     columns={
#     "all_names": "sentence2",
#     "concept_id": "concept_id2"}
#     )
    
#     neg_df.columns
#     # ['index', 'concept_id2', 'sentence2']
    
    
#     ## merge with the original negative_samples_list
#     negative_samples = negative_samples.merge(
#         neg_df[['index', 'concept_id2', 'sentence2']],
#         left_on='index',
#         right_on='index',
#         how='inner'
#     ).drop(columns=['choices']).reset_index(drop=True)
    
#     negative_samples['label'] = 0
#     negative_samples['source'] = 'matching_negative'
    
#     negative_samples = negative_samples[['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label', 'source']]
    
#     negative_samples.columns
#     # ['sentence1', 'concept_id1', 'index', 'concept_id2', 'sentence2', 'source']
    
#     return negative_samples


class MatchingNegativeIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, 
                 positive_dataset_matching, 
                 concept_ancestor, 
                 std_target_for_matching, 
                 n_neg=4, 
                 seed=42):
        """
        Iterable dataset for generating negative samples dynamically.
        
        Args:
            positive_dataset_matching (Iterable): Iterable dataset of positive pairs.
            concept_ancestor (pd.DataFrame): Concept ancestor relationships.
            std_target_for_matching (pd.DataFrame): Standard concept dataset.
            n_neg (int): Number of negative samples per positive sample.
            seed (int): Random seed.
        """
        print("get in MatchingNegativeIterableDataset")
        if isinstance(positive_dataset_matching, pd.DataFrame):
            self.positive_dataset_matching = positive_dataset_matching.to_dict(orient="records")
        else:
            self.positive_dataset_matching = positive_dataset_matching  # Already iterable
        self.concept_ancestor = concept_ancestor
        self.std_target_for_matching = std_target_for_matching
        self.n_neg = n_neg
        self.seed = seed
        random.seed(seed)

        # Special token for sentence pairs
        self.matching_token = "[MATCHING]"

        # Precompute exclusion lists (relation maps)
        target_ids_for_matching = std_target_for_matching['concept_id']
        self.concept_ancestor_filtered = get_filtered_concept_ancestor(concept_ancestor, target_ids_for_matching)
        relation_maps = create_relation_maps(self.concept_ancestor_filtered)
        
        relation_maps2 = relation_maps[['from_concept_id', 'to_concept_id']].copy()
        relation_maps2.set_index("from_concept_id", inplace=True)
        relation_maps2['to_concept_id'] = relation_maps2['to_concept_id'].apply(lambda x: set(x))
        self.std_id_to_excluded = relation_maps2.to_dict()['to_concept_id']
        
        # Prepare concept mapping for negatives
        std_maps_df = std_target_for_matching[['concept_id', "all_nonstd", "std_name"]].copy()
        std_maps_df["all_names"] = std_maps_df.apply(lambda x: x["all_nonstd"] + [x["std_name"]], axis=1)
        std_maps_df['maps_num'] = std_maps_df['all_names'].apply(len)
        
        self.id_to_names_df = std_maps_df[['concept_id', 'all_names']].copy()
        self.id_to_names_df['sample_idx'] = self.id_to_names_df['all_names'].apply(lambda x: list(range(len(x))))
        self.id_to_names_df = self.id_to_names_df.explode(["all_names", 'sample_idx'])

        self.all_std_ids = std_target_for_matching["concept_id"].to_list()
        self.all_std_ids_set = set(self.all_std_ids)

    def _generate_negative_samples(self, row):
        """Generate n_neg negative samples dynamically for a given row"""
        std_id = row["concept_id1"]
        exclude_ids = self.std_id_to_excluded.get(std_id, set())

        # Randomly sample 2*n_neg candidates
        choices = random.sample(self.all_std_ids, 2 * self.n_neg)
        choices_filtered = list(set(choices) - exclude_ids)

        # Adjust number of negatives
        if len(choices_filtered) < self.n_neg:
            n_neg_more = self.n_neg - len(choices_filtered)
            candidate_ids = list(self.all_std_ids_set - exclude_ids - set(choices_filtered))
            additional_ids = random.sample(candidate_ids, min(n_neg_more, len(candidate_ids)))
            choices_filtered += additional_ids
        elif len(choices_filtered) > self.n_neg:
            choices_filtered = choices_filtered[:self.n_neg]

        # Assign negative names
        for neg_id in choices_filtered:
            filtered_df = self.id_to_names_df[self.id_to_names_df["concept_id"] == neg_id]
            if not filtered_df.empty:
                mapped_name = filtered_df["all_names"].sample().values[0]
                yield {
                "sentence1": self.matching_token + " " + row["sentence1"],
                "sentence2": self.matching_token + " " + mapped_name, 
                "concept_id1": row["concept_id1"],
                "concept_id2": neg_id,
                "label": 0,
                "source": "matching_negative"
                }
            

    def __iter__(self):
        """Yields only negative samples dynamically"""
        for row in self.positive_dataset_matching:
            yield from self._generate_negative_samples(row) 


def generate_relation_positive_samples(
    concept_ancestor_filtered: pd.DataFrame,
    concept: pd.DataFrame
):
    """
    Generate positive samples for ancestor-descendant relationships.
    
    Args:
        concept_ancestor_filtered (pd.DataFrame): Filtered concept_ancestor table that you want to use to generate positive samples.
        concept (pd.DataFrame): Concept table to provide concept names.
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


def add_special_token(vec, token):
    vec = token + " " + vec
    return vec


def add_special_token_df(df, token):
    df['sentence1'] = add_special_token(df['sentence1'], token)
    df['sentence2'] = add_special_token(df['sentence2'], token)
    return df