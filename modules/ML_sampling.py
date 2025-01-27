import pandas as pd
import random
from tqdm import tqdm



def create_relation_maps(concept_ancestor, target_ids):
    """
    Create a mapping between standard concepts and their parents/children.

    Args:
        concept_ancestor (pd.DataFrame): The concept_ancestor table.
        target_ids (pd.DataFrame): Which standard concepts to consider.
        
    Returns:
        pd.DataFrame: A DataFrame with the mapping between standard concepts and their parents/children
        columns: ['from_concept_id', 'to_concept_id', 'min_levels_of_separation', 'max_levels_of_separation', 'type'
    
    """
    ## keep those that are in the target concept list
    ## Remove the concept that maps to itself
    concept_ancestor_filtered = concept_ancestor[
        (
            concept_ancestor['ancestor_concept_id'].isin(target_ids)|
            concept_ancestor['descendant_concept_id'].isin(target_ids)
        )&
        concept_ancestor['min_levels_of_separation']!=0
        ]

    ## For a given standard concept, create a mapping to its children
    concept_ancestor_map = concept_ancestor_filtered[
        concept_ancestor_filtered['ancestor_concept_id'].isin(target_ids)
    ].rename(
            columns={
            'ancestor_concept_id': 'from_concept_id',
            'descendant_concept_id': 'to_concept_id'
    })

    ## For a given standard concept, create a mapping to its parents
    concept_offspring_map = concept_ancestor_filtered[
        concept_ancestor_filtered['descendant_concept_id'].isin(target_ids)
    ].rename(
            columns={
            'descendant_concept_id': 'from_concept_id',
            'ancestor_concept_id': 'to_concept_id'
    })

    concept_ancestor_map['type']='ancestor'
    concept_offspring_map['type']='offspring'

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
def generate_positive_samples(df, columns, column_ids):
    """
    Create a dataset that contains 1-1 mappings between standard concepts and non-standard concepts.

    Args:
        df (pd.DataFrame): The input DataFrame with at least columns ['std_name', 'concept_id']
        columns (list of str): The columns to process (e.g., ['nonstd_name', 'synonym_name', 'descriptions']).
        column_ids (list of str): The corresponding columns with concept IDs (e.g., ['nonstd_concept_id', None, None]).

    Returns:
        pd.DataFrame: A processed dataset with exploded rows and additional metadata.
    """
    column_keep = ['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label1', 'source']
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
        exploded_df['label1'] = 1
        exploded_df = exploded_df[column_keep]
        ## save as excel for inspection
        ## exploded_df.to_excel(f'positive_samples_{column}.xlsx', index=False)
        result_frames.append(exploded_df)
        
    final_dataset = pd.concat(result_frames, ignore_index=True).drop_duplicates()
    return final_dataset


# positive_samples = positive_sample_dataset

def generate_negative_samples(
    positive_samples: pd.DataFrame,
    relation_maps: pd.DataFrame,
    std_target_with_nonstd: pd.DataFrame,
    n_neg: int = 4,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate negative samples for each row in positive_samples.
    
    Parameters
    ----------
    positive_samples : pd.DataFrame
        Must contain at least columns: ["sentence1", "concept_id1"].
    relation_maps : pd.DataFrame
        includes columns ["from_concept_id", "to_concept_id"] 
        where to_concept_id is a list of parent/child for that from_concept_id.
    std_target_with_nonstd : pd.DataFrame
        Must contain ["concept_id", "all_nonstd", "std_name"] for standard concepts.
    n_neg : int
        Number of negative samples to generate per positive sample row.
    seed : int
        Random seed for reproducible sampling.
        
    Returns
    -------
    pd.DataFrame
        Columns: ["sentence1", "sentence2", "concept_id1", "concept_id2", "label1", "source"] 
        with label1 = 0 for negatives, source = "negative".
    """
    random.seed(seed)
    
    # Make a dict to map std concept id to its exclusion list
    relation_maps2 = relation_maps[['from_concept_id', 'to_concept_id']].copy()
    relation_maps2.set_index("from_concept_id", inplace=True)
    relation_maps2['to_concept_id'] = relation_maps2['to_concept_id'].apply(lambda x: set(x))
    std_id_to_excluded = relation_maps2.to_dict()['to_concept_id']
    
    ## prepare the mapping from concept_id to all possible names
    std_maps_df = std_target_with_nonstd[['concept_id', "all_nonstd", "std_name"]].copy()
    std_maps_df["all_names"] = std_maps_df.apply(
        lambda x: x["all_nonstd"] + [x["std_name"]], axis=1
    )
    std_maps_df['maps_num'] = std_maps_df['all_names'].apply(lambda x: len(x))
    
    ## Column: concept_id, all_names, sample_idx
    id_to_names_df=std_maps_df[['concept_id', 'all_names']].copy()
    id_to_names_df['sample_idx'] = id_to_names_df['all_names'].apply(lambda x: [i for i in range(len(x))])
    id_to_names_df = id_to_names_df.explode(
    ["all_names", 'sample_idx']
    )
    
    
    # for each row in positive_samples:
    # 1. randomly sample 2*n_neg standard concept IDs
    # 2. filter out the ones in the exclusion list
    # 3. keep n_neg samples from the remaining set
    all_std_ids = std_target_with_nonstd["concept_id"].to_list()
    all_std_ids_set = set(all_std_ids)
    
    # 1. randomly sample 2*n_neg standard concept IDs
    negative_samples = positive_samples[['sentence1', 'concept_id1']].copy()
    negative_samples['index'] = negative_samples.index
    negative_samples["choices"] = [random.sample(all_std_ids, 2*n_neg) for _ in range(len(positive_samples))]
    
    ## Refine the sampled choices such that they are not in the exclude set
    for i, row in tqdm(negative_samples.iterrows(), total=len(negative_samples)):
        std_id = row["concept_id1"]
        choices = row["choices"]
        exclude_ids = std_id_to_excluded.get(std_id, set())
        choices_filtered = list(set(choices) - exclude_ids)
        
        ## if the filtered set is too small, add more
        ## if the filtered set is too large, keep the first n_neg
        if len(choices_filtered) < n_neg:
            n_neg_more = n_neg - len(choices_filtered)
            candidate_ids = list(all_std_ids_set - exclude_ids - set(choices_filtered))
            n_neg_more = min(n_neg_more, len(candidate_ids))
            additional_ids = random.sample(candidate_ids, n_neg_more)
            choices_filtered += additional_ids
        elif len(choices_filtered) > n_neg:
            choices_filtered = choices_filtered[:n_neg]
        
        ## update the choices
        negative_samples.at[i, "choices"] = choices_filtered
        
    
    neg_df = negative_samples[['index','choices']]
    neg_df = neg_df.explode("choices")
    ## randomly pick number from 0 to 100000 with replacement
    neg_df['sample_idx'] = [random.randint(0, 100000) for _ in range(len(neg_df))]
    
    # Obtain the number of possible negative samples for each concept_id
    neg_df = neg_df.merge(
        std_maps_df[['concept_id', 'maps_num']],
        left_on='choices',
        right_on='concept_id',
        how='left'
    ).drop(columns='concept_id')
    
    neg_df.columns
    # ['index', 'choices', 'sample_idx', 'maps_num']
    
    ## recalculate the sample_idx to be within the range of maps_num
    neg_df['sample_idx'] = neg_df.apply(
        lambda x: x['sample_idx'] % x['maps_num'], axis=1
    )
    len(neg_df) # 2825424
    
    ## obtain the name of the mapped nonstd name
    neg_df = neg_df.merge(
        id_to_names_df,
        left_on=['choices', 'sample_idx'],
        right_on=['concept_id', 'sample_idx'],
        how='inner'
    ).drop(
        columns=['maps_num', 'choices', 'sample_idx']
    ).rename(
    columns={
    "all_names": "sentence2",
    "concept_id": "concept_id2"}
    )
    
    neg_df.columns
    # ['index', 'concept_id2', 'sentence2']
    
    
    ## merge with the original negative_samples_list
    negative_samples = negative_samples.merge(
        neg_df[['index', 'concept_id2', 'sentence2']],
        left_on='index',
        right_on='index',
        how='inner'
    ).drop(columns=['choices']).reset_index(drop=True)
               
    negative_samples['source'] = 'negative'
    
    negative_samples.columns
    # ['sentence1', 'concept_id1', 'index', 'concept_id2', 'sentence2', 'source']
    
    return negative_samples
