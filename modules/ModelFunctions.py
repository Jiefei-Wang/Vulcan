import json
import os
import datetime
import shutil
import re
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from modules.TOKENS import TOKENS

## add all special tokens
special_tokens = TOKENS.all_tokens

def get_base_model(base_model, special_tokens):
    model = SentenceTransformer(base_model)

    ## add special tokens
    transformer = model[0]  
    tokenizer = transformer.tokenizer
    auto_model = transformer.auto_model

    # Now you can add special tokens to the tokenizer
    special_tokens_dict = {
        "additional_special_tokens": special_tokens
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    # Resize the model embeddings if we added tokens
    if num_added_tokens > 0:
        auto_model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def save_init_model(model, tokenizer, save_folder):
    """
    Automatically saves the model and tokenizer to the init subdirectory of a given folder 

    Args:
        model: The model to save (assuming it has a `save` method).
        tokenizer: The tokenizer to save (assuming it has a `save_pretrained` method).

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    init_folder = os.path.join(save_folder, 'init')
    if not os.path.exists(init_folder):
        try:
            os.makedirs(init_folder)
        except OSError as e:
            print(f"Error creating save folder: {e}")
            return False  # Indicate error
    
    try:
        # Save the model and tokenizer
        model.save(init_folder)  
        tokenizer.save_pretrained(init_folder) 
        print(f"Model and tokenizer saved to {init_folder}")
    except Exception as e:
        print(f"Error saving model or tokenizer: {e}")
        return False  # Indicate error
    
    return True

def save_best_model(model, tokenizer, save_folder, train_config = {}):
    save_path = os.path.join(save_folder, "best_model")
    model.save(save_path)  # Assuming model has a save method
    tokenizer.save_pretrained(save_path)  # Save tokenizer as well
    with open(os.path.join(save_path, 'train_config.json'), 'w') as f:
        json.dump(train_config, f)



def auto_load_model(model_path_or_name):
    """
    Automatically loads a model and tokenizer from either a local path or HuggingFace model name.

    For local paths:
    - If the path exists and contains checkpoint- models, loads the latest checkpointed model
    - If the path exists, has no checkpoint- models, but init exists, loads init as a SentenceTransformer model
    - If the path exists but has no checkpoint- models, no init, loads directly as a SentenceTransformer model
    - If the path exists but loading fails, returns (None, None)
    - If the path does not exist, load it as a HuggingFace model name


    Args:
        model_path_or_name (str): Either a local path to saved models or a HuggingFace model name.

    Returns:
        tuple: A tuple containing (model, tokenizer).
               Returns (None, None) if loading fails.
    """

    if os.path.exists(model_path_or_name):
        saved_models = [f for f in os.listdir(model_path_or_name) if f.startswith("checkpoint-")]

        if saved_models:
            # Extract indices and sort by index to find the latest model.
            models_with_indices = []
            for filename in saved_models:
                match = re.match(r"checkpoint-(\d+)", filename)
                if match:
                    try:
                        index = int(match.group(1))
                        models_with_indices.append((index, filename))
                    except ValueError:
                        print(f"Warning: Could not parse index from filename: {filename}")

            if not models_with_indices:
                return None, None

            # Find the model with the largest index (latest model).
            latest_index = max(index for index, filename in models_with_indices)
            latest_model_filename = next(filename for index, filename in models_with_indices if index == latest_index)
            target_model_path = os.path.join(model_path_or_name, latest_model_filename)

        elif os.path.exists(os.path.join(model_path_or_name, 'init')):
            target_model_path = os.path.join(model_path_or_name, 'init')
        else:
            print("No auto-saved models found, loading as direct SentenceTransformer model...")
            target_model_path = model_path_or_name
            
        
        try:
            # Load the model and tokenizer
            model = SentenceTransformer(target_model_path)
            tokenizer = AutoTokenizer.from_pretrained(target_model_path)
            print(f"Loaded latest auto-saved model from: {target_model_path}")
            return model, tokenizer
        except Exception as e:
            print(f"Error loading auto-saved model from {target_model_path}: {e}")
            return None, None
            
            


def get_loss(loss_func, block_tokenizer, idx):
    data_block = block_tokenizer[idx]
    # --- Process the Data Block ---
    labels_tensor = data_block['labels']
    tokenized_sentence1s = data_block['tokenized_sentence1s']
    tokenized_sentence2s = data_block['tokenized_sentence2s']
    
    sentence_pairs = [tokenized_sentence1s, tokenized_sentence2s]

    # Forward pass
    loss_value = loss_func(sentence_pairs, labels_tensor)
    return loss_value





def get_ST_model(base_model = 'ClinicalBERT'):
    """
    Load a SentenceTransformer model or create a new one if it doesn't exist. The model will be saved and reused in the future.
    """
    token_combined = '_'.join(special_tokens)
    # keep only letters, numbers, and underscores
    token_combined = re.sub(r'\W+', '', token_combined)
    
    base_model_path = f'models/{base_model}'
    saved_path = f'models/{base_model}_ST_{token_combined}'
    if not os.path.exists(saved_path):
        model, tokenizer = get_base_model(base_model_path, special_tokens)
        save_init_model(model, tokenizer, saved_path, max_saves=1) 
    else:
        model, tokenizer, _ = auto_load_model(saved_path)
    return model, tokenizer



def encode_concepts(model, concepts, normalize_embeddings=True):
    """
    Encodes a list of concepts using the provided model. Handle duplicates
    """
    if isinstance(concepts, list):
        concepts = np.array(concepts) 

    unique_concepts = np.unique(concepts)
    embeddings = model.encode(
        unique_concepts.tolist(), 
        normalize_embeddings=normalize_embeddings
    )
    concept_to_embedding = dict(zip(unique_concepts, embeddings))
    concept_embeddings = np.array([concept_to_embedding[concept] for concept in concepts])
    return concept_embeddings
