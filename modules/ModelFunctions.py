import json
import os
import datetime
import shutil
import re

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from modules.TOKENS import TOKENS

special_tokens = [TOKENS.child]

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


def get_model_indices(path):
    """
    Get the indices of saved models in the specified path.

    Args:
        path (str): The path to the folder where models are saved.

    Returns:
        list: A list of indices of saved models.
    """
    saved_models = [f for f in os.listdir(path) if f.startswith("auto_save_")]
    indices = []
    for filename in saved_models:
        match = re.match(r"auto_save_(\d+)_", filename)  # Use regex to extract the index
        if match:
            try:
                indices.append(int(match.group(1)))
            except ValueError:
                pass
            
    # Sort indices and its corresponding filenames
    filenames = [x for _, x in sorted(zip(indices, saved_models))]
    indices = sorted(indices)
    return indices, filenames

def auto_save_model(model, tokenizer, save_folder, max_saves, train_config = {}):
    """
    Automatically saves the model and tokenizer to a folder with a patterned name,
    managing the number of saved models by deleting older ones when necessary.

    Args:
        model: The model to save (assuming it has a `save` method).
        tokenizer: The tokenizer to save (assuming it has a `save_pretrained` method).
        save_folder (str): The path to the folder where models will be saved.
        max_saves (int): The maximum number of saved models to keep.

    Returns:
        int: The updated current_save_index. Returns -1 if there's an error.
    """

    if not os.path.exists(save_folder):
        try:
            os.makedirs(save_folder)
        except OSError as e:
            print(f"Error creating save folder: {e}")
            return -1  # Indicate error

    model_indices, _ = get_model_indices(save_folder)
    if not model_indices:
        current_save_index = 1
    else:
        current_save_index = max(model_indices) + 1  # Next available index

    save_name = f"auto_save_{current_save_index}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join(save_folder, save_name)

    try:
        # Save the model and tokenizer
        model.save(save_path)  # Assuming model has a save method
        tokenizer.save_pretrained(save_path)  # Save tokenizer as well
        with open(os.path.join(save_path, 'train_config.json'), 'w') as f:
            json.dump(train_config, f)
        print(f"Model and tokenizer saved to {save_path}")
    except Exception as e:
        print(f"Error saving model or tokenizer: {e}")
        return -1  # Indicate error

    # Manage the number of saved models
    
    model_indices, file_names = get_model_indices(save_folder)

    error_num = 0
    while len(model_indices) > max_saves:
        file_name = file_names[0]  # oldest based on filename index
        oldest_model_path = os.path.join(save_folder, file_name)
        try:
            shutil.rmtree(oldest_model_path)
            print(f"Deleted oldest model: {oldest_model_path}")
        except Exception as e:
            print(f"Error deleting oldest model {oldest_model_path}: {e}")
            import subprocess
            subprocess.call(['powershell', '-Command', f'Remove-Item -Path "{oldest_model_path}" -Recurse -Force'])
            error_num += 1
            if error_num > 2:
                print("Error deleting old models, stopping auto-delete.")
                break
            
        model_indices, file_names = get_model_indices(save_folder)

    return current_save_index

def save_best_model(model, tokenizer, save_folder, train_config = {}):
    save_path = os.path.join(save_folder, "best_model")
    model.save(save_path)  # Assuming model has a save method
    tokenizer.save_pretrained(save_path)  # Save tokenizer as well
    with open(os.path.join(save_path, 'train_config.json'), 'w') as f:
        json.dump(train_config, f)



def auto_load_model(save_folder):
    """
    Automatically loads the latest saved model and tokenizer from the specified folder.

    Args:
        save_folder (str): The path to the folder where models are saved.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer, or (None, None) if no model is found or loading fails.
    """

    if not os.path.exists(save_folder):
        print(f"Save folder does not exist: {save_folder}")
        return None, None

    saved_models = [f for f in os.listdir(save_folder) if f.startswith("auto_save_")]

    if not saved_models:
        print("No auto-saved models found in the specified folder.")
        return None, None

    # Extract indices and sort by index to find the latest model.
    models_with_indices = []
    for filename in saved_models:
        match = re.match(r"auto_save_(\d+)_", filename)
        if match:
            try:
                index = int(match.group(1))
                models_with_indices.append((index, filename))
            except ValueError:
                print(f"Warning: Could not parse index from filename: {filename}")

    if not models_with_indices:
        print("No auto-saved models with valid index found in the specified folder.")
        return None, None

    # Find the model with the largest index (latest model).
    latest_index = max(index for index, filename in models_with_indices)
    latest_model_filename = next(filename for index, filename in models_with_indices if index == latest_index)
    latest_model_path = os.path.join(save_folder, latest_model_filename)

    try:
        # Load the model and tokenizer
        model = SentenceTransformer(latest_model_path)  # Load the model
        tokenizer = AutoTokenizer.from_pretrained(latest_model_path)  # Load tokenizer
        train_config_path = os.path.join(latest_model_path, 'train_config.json')
        if os.path.exists(train_config_path):
            with open(train_config_path, 'r') as f:
                train_config = json.load(f)
        else:
            train_config = {}
        print(f"Loaded latest model and tokenizer from: {latest_model_path}")
        return model, tokenizer, train_config
    except Exception as e:
        print(f"Error loading model or tokenizer from {latest_model_path}: {e}")
        return None, None, None


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
        auto_save_model(model, tokenizer, saved_path, max_saves=1) 
    else:
        model, tokenizer, _ = auto_load_model(saved_path)
    return model, tokenizer