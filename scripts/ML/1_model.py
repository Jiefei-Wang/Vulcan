## create a sentence transformer model and tokenizer from a base model
from modules.ML_train import get_base_model, auto_save_model
from modules.TOKENS import TOKENS
special_tokens = [TOKENS.child]

base_model = 'ClinicalBERT'
base_model_path = f'models/{base_model}'
saved_path = f'models/{base_model}_ST'
model, tokenizer = get_base_model(base_model_path, special_tokens)
auto_save_model(model, tokenizer, saved_path, max_saves=1) 