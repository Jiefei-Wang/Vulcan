import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import math
import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import pandas as pd
from datetime import datetime
import wandb
from transformers import get_linear_schedule_with_warmup 
from sentence_transformers import losses
from modules.ML_data import get_matching, get_relation, get_matching_validation,get_relation_positive_validation, DictBatchSampler
from modules.ML_train import get_base_model, auto_save_model, save_best_model, CustomEvaluator
from modules.TOKENS import TOKENS
from tqdm.auto import tqdm 

from modules.timed_logger import logger
logger.reset_timer()


# ## disable default huggingface logging
# logging.getLogger("transformers").setLevel(logging.WARNING)  # Or logging.ERROR
# logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


## make sure cuda is available
torch.cuda.is_available()


base_model = 'all-MiniLM-L6-v2'
base_model_path = f'models/{base_model}'
# base_model = 'models/base_exclude_CIEL/checkpoint-65000'
output_dir = f"output/{base_model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

special_tokens = [TOKENS.child]
## what datasets to use?
ds_names = ['matching', 'ancestor']

## columns to use
cols = ['sentence1', 'sentence2', 'label']


#################################
## For Training
#################################
logger.log("Loading training data")
n_neg_matching = 50
n_neg_relation = 50
n_fp_matching = 50
dt_seed = 42
data_folder = 'data/ML'
iterable_matching = get_matching(data_folder, n_neg=n_neg_matching, seed=dt_seed, n_fp=n_fp_matching)
iterable_offspring, iterable_ancestor = get_relation(data_folder, n_neg = n_neg_relation, seed=dt_seed)


#################################
## For validation
#################################
logger.log("Loading validation data")
reserved_concepts = pd.read_feather("data/ML/base_data/reserved_concepts.feather")
positive_df_matching = pd.read_feather('data/ML/matching/positive_df_matching.feather')

reserved_concepts.columns
# ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
#        'concept_class_id', 'standard_concept', 'concept_code',
#        'valid_start_date', 'valid_end_date', 'invalid_reason',
#        'std_concept_id']

nonstd_conditions = reserved_concepts[reserved_concepts['domain_id'] == 'Condition'].reset_index(drop=True)

# all standard concepts used in positive_df_matching
database = positive_df_matching[['concept_id1', 'sentence1']].drop_duplicates().rename(
    columns={'concept_id1': 'concept_id', 'sentence1': 'concept_name'})

query = nonstd_conditions[['concept_id', 'concept_name', 'std_concept_id']]




#################################
## Model
#################################
logger.log("Loading model")
model, tokenizer = get_base_model(base_model_path, special_tokens)

evaluator = CustomEvaluator()




mini_batch_size = 64 
arg_eval_steps = 1024
arg_saving_steps = 1024*2

fp16 = True # Use Mixed Precision (Float16)
warmup_ratio = 0.1
learning_rate = 2e-5 # Typical default for AdamW with transformers

max_saves = 4
epoch_num = 5
best_eval_loss = float('inf')

# --- Initialization ---
wandb.init(project="Concept_Mapping", name=output_dir)

# 1. Model and Tokenizer
model, tokenizer = get_base_model(base_model_path, special_tokens)

# 2. Evaluator
evaluator = CustomEvaluator()

# 3. Loss Function
train_loss = losses.ContrastiveLoss(model=model)

# 4. Device Handling
device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
model = model.to(device)
train_loss = train_loss.to(device)
print(f"Using device: {device}")

# 5. Data Sampler
train_dataset_dict = {
    'matching': iterable_matching,
    'offspring': iterable_offspring,
    'ancestor': iterable_ancestor
}
train_dataset_dict = {k: train_dataset_dict[k] for k in ds_names}

ratios = {'matching': 1, 'offspring': 0.5, 'ancestor': 0.5}
ratios = {k: ratios[k] for k in ds_names}
# Determine block_size and shuffle_buffer 
# These affect how much data the sampler loads/shuffles at once
# The sampler yields blocks, we process them with mini-batches inside
shuffle_buffer = mini_batch_size * arg_saving_steps * 2 # As per your code
sampler_batch_size = mini_batch_size * 64
sampler = DictBatchSampler(train_dataset_dict, batch_size=sampler_batch_size, ratios=ratios, shuffle_buffer=shuffle_buffer)
ds_sizes = sampler.element_size()
iterations = sampler.iteration_size()
iterations_per_epoch = max(iterations.values()) if iterations else 1 # Avoid division by zero

print(f"Data Sizes: {ds_sizes}")
print(f"Sampler Iterations: {iterations}")
print(f"Iterations per Epoch: {iterations_per_epoch}")

# 6. Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 7. Scheduler
# Calculate total steps based on your epoch definition
num_training_steps = int(epoch_num * iterations_per_epoch * sampler_batch_size / mini_batch_size)
num_warmup_steps = int(num_training_steps * warmup_ratio)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# 8. Mixed Precision Scaler
scaler = GradScaler(device=device_type, enabled=fp16)

# --- Manual Training Loop ---
global_step = 0
current_epoch = -1
epoch_loss = 0.0
wandb.watch(model) # Log gradients and model topology


progress_bar_outer = tqdm(range(num_training_steps), desc="Mini Steps") # Tracks overall progress

for j, data_block in enumerate(sampler):
    epoch_i = j // iterations_per_epoch
    if epoch_i >= epoch_num:
        print(f"Target epochs ({epoch_num}) reached. Stopping training.")
        break

    if epoch_i != current_epoch:
        print(f"\n--- Starting Epoch {epoch_i + 1}/{epoch_num} ---")
        current_epoch = epoch_i
        epoch_loss = 0.0
        
        model.eval()
        with torch.no_grad():
            evaluator.build_reference(model, database)

    # --- Process the Data Block ---
    model.train() # Set model to training mode for this block

    # Determine number of mini-batches within this block
    # Assuming data_block is a list of examples
    num_examples_in_block = len(data_block)
    num_mini_batches = math.ceil(num_examples_in_block / mini_batch_size)
    sentence1s = data_block['sentence1']
    sentence2s = data_block['sentence2']
    labels = data_block['label']
    ## labels to tensor
    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    
    
    # Process the block in mini-batches
    for i in range(num_mini_batches):
        start_idx = i * mini_batch_size
        end_idx = start_idx + mini_batch_size
        mini_batch = data_block[start_idx:end_idx]
        
        mini_sentence1s = sentence1s[start_idx:end_idx]
        mini_sentence2s = sentence2s[start_idx:end_idx]
        mini_labels = labels[start_idx:end_idx]
        

        if not mini_batch:
            raise ValueError(f"Empty mini_batch at index {i} in block {j}.")

        tokenized_sentence1s = tokenizer(
            mini_sentence1s,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        tokenized_sentence2s = tokenizer(
            mini_sentence2s,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        mini_sentence_pairs = [tokenized_sentence1s, tokenized_sentence2s]
        

        # Forward pass
        optimizer.zero_grad()
        with autocast(device_type, enabled=fp16):
            loss_value = train_loss(mini_sentence_pairs, mini_labels)


        # Backward pass
        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() # Step scheduler after optimizer step
        loss_value
        
        epoch_loss += loss_value.item()
        epoch_avg_loss = epoch_loss / (j % iterations_per_epoch + 1)
        wandb.log({
            "train/loss": loss_value.item(),
            "train/learning_rate": scheduler.get_last_lr()[0],
            "epoch": current_epoch + (j % iterations_per_epoch) / iterations_per_epoch,
            "epoch_avg_loss": epoch_avg_loss
        }, step=global_step)

        global_step += 1
        progress_bar_outer.update(1)
        progress_bar_outer.set_postfix({"Loss": f"{loss_value.item():.4f}", "Epoch": f"{current_epoch+1}", "Epoch Avg Loss": f"{epoch_avg_loss:.4f}"})
        
        ## save model every arg_saving_steps
        if global_step % arg_saving_steps == 0:
            print("Saving checkpoint...")
            auto_save_model(model, tokenizer, output_dir, max_saves=max_saves)
        

    # Log training loss periodically
    if global_step % arg_eval_steps == 0:
        model.eval()
        with torch.no_grad():
            eval_metrics = evaluator(query, n_results=100, k_list= [1, 10, 50])
            eval_results = {f"eval/{k}": v for k, v in eval_metrics.items()}
            wandb.log(eval_results, step=global_step)

        eval_loss = eval_metrics['top 1']
        # 4. Save Best Model
        if eval_loss < best_eval_loss:
            print(f"New best model found! Loss improved from {best_eval_loss:.4f} to {eval_loss:.4f}. Saving...")
            best_eval_loss = eval_loss
            save_best_model(model, tokenizer, output_dir) # Pass necessary args
            wandb.log({"eval/best_loss": best_eval_loss}, step=global_step)



# --- End of Training ---
progress_bar_outer.close()
wandb.finish()
logger.done()