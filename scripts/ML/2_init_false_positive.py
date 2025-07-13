from modules.FalsePositives import get_false_positives
import os
import pandas as pd
from modules.ML_train import auto_load_model


n_fp_matching = 50
base_path = "data/matching"
std_condition_concept = pd.read_feather(os.path.join(base_path, 'std_condition_concept.feather'))

target_concepts = std_condition_concept

base_model = 'ClinicalBERT'
ST_model_path = f'models/{base_model}_ST'
model, tokenizer = auto_load_model(ST_model_path)

fp = get_false_positives(model, target_concepts,  n_fp_matching=n_fp_matching)
fp.to_feather(os.path.join(base_path, f'fp_matching_{n_fp_matching}.feather'))
