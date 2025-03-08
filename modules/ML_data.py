import os
import pandas as pd
from modules.ML_sampling import AncestorIterableDataset, MatchingIterableDataset, OffspringIterableDataset



def get_matching(data_folder, n_neg=4, seed=42):
    positive_dataset_matching = pd.read_feather(
        os.path.join(data_folder, 'matching/positive_dataset_matching.feather')
        )
    candidate_df_matching = pd.read_feather(
        os.path.join(data_folder, 'matching/candidate_dataset_matching.feather')
        )
    

    iterable_matching = MatchingIterableDataset(
        positive_dataset_matching,
        candidate_df_matching,
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