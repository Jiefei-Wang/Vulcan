from modules.ModelFunctions import save_init_model, get_base_model
from modules.TOKENS import TOKENS

tokens = [TOKENS.parent]
base_model = 'all-MiniLM-L6-v2'
model, tokenizer = get_base_model(base_model, tokens)
save_init_model(model, tokenizer, f'output/{base_model}')