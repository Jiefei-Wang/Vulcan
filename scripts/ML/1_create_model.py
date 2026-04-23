from modules.ModelFunctions import save_init_model, get_base_model, get_ST_model
from modules.TOKENS import TOKENS


tokens = [TOKENS.parent]
model_name = 'biobert-base-cased-v1.2'
model, tokenizer = get_base_model(f"models/{model_name}", tokens)
# save_init_model(model, tokenizer, f'output/{model_name}')

get_ST_model(model_name, tokens)