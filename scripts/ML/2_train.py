import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import math

import numpy as np
import pandas as pd
import wandb
import tempfile
import torch
from tqdm import tqdm
from datasets import Dataset
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from datetime import datetime
from transformers import get_linear_schedule_with_warmup 
from sentence_transformers import SentenceTransformerTrainer, losses

from modules.ML_train import auto_load_model, auto_save_model, save_best_model
from modules.BlockTokenizer import BlockTokenizer
from modules.TOKENS import TOKENS
from modules.timed_logger import logger
from modules.Dataset import PositiveDataset, NegativeDataset, CombinedDataset
logger.reset_timer()
logger.log("")


special_tokens = [TOKENS.child]


# --- Initialization ---
output_dir = f"output/finetune/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
base_model = 'ClinicalBERT'
ST_model_path = f'models/{base_model}_ST'

#################################
## For Training
#################################
logger.log("Loading training data")
n_pos_matching = 10
n_neg_matching = 50
n_neg_relation = 50
n_fp_matching = 50
dt_seed = 42
base_path = "data/matching"
std_condition_concept = pd.read_feather(os.path.join(base_path, 'std_condition_concept.feather'))
condition_matching_name_bridge_train = pd.read_feather(os.path.join(base_path, 'condition_matching_name_bridge_train.feather'))
condition_matching_name_table_train = pd.read_feather(os.path.join(base_path, 'condition_matching_name_table_train.feather'))

matching_pos = PositiveDataset(
    target_concepts=std_condition_concept,
    name_table=condition_matching_name_table_train,
    name_bridge=condition_matching_name_bridge_train,
    max_elements=n_pos_matching
)

matching_neg = NegativeDataset(
    target_concepts=std_condition_concept,
    name_table=condition_matching_name_table_train,
    blacklist_bridge=condition_matching_name_bridge_train,
    max_elements=n_neg_matching
)

ds_all = CombinedDataset(
    positive= matching_pos,
    negative= matching_neg
)

#################################
## For validation
#################################
logger.log("Loading validation data")
if False:
    condition_matching_valid = pd.read_feather(os.path.join(base_path, 'condition_matching_valid.feather'))

    matching_valid = condition_matching_valid[['sentence1', 'sentence2']].copy()
    matching_valid['label'] = 1
    matching_valid = Dataset.from_pandas(matching_valid)


#################################
## Model
#################################
logger.log("Loading model")
fp16 = True # Use Mixed Precision (Float16)
learning_rate = 2e-5 # Typical default for AdamW with transformers
model, tokenizer = auto_load_model(ST_model_path)
# evaluator = CustomEvaluator()
max_saves = 4

# --- Initialization ---
wandb_report_steps = 256
temp_dir = tempfile.gettempdir()
wandb.init(project="Concept_Mapping", name=output_dir, dir = temp_dir)

# Loss and Optimizer
train_loss = losses.ContrastiveLoss(model=model)
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 4. Device Handling
device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
model = model.to(device)
train_loss = train_loss.to(device)
print(f"Using device: {device}")


# 7. Scheduler
epoch_num = 1
buffer_size = 256 * 16
batch_size = 256
arg_eval_steps = 2048
arg_saving_steps = arg_eval_steps*2


block_tokenizer = BlockTokenizer(
    dataset = ds_all, 
    buffer_size = buffer_size, 
    batch_size= batch_size, 
    tokenizer = tokenizer, 
    device = device)

warmup_ratio = 0.1
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps= len(ds_all) // batch_size * warmup_ratio,
    num_training_steps= len(ds_all) // batch_size * epoch_num
)

# 8. Mixed Precision Scaler
scaler = GradScaler(device=device_type, enabled=fp16)

# --- Manual Training Loop ---
global_step = 0
best_eval_accuracy = float('-inf')
wandb.watch(model) # Log gradients and model topology

progress_bar = tqdm(total=len(block_tokenizer) * epoch_num, desc="Training Progress", unit="batch")
for epoch_i in range(epoch_num):
    # evaluator.build_reference(model, std_condition_concept)  
    epoch_total_loss = 0.0
    for batch_i in range(len(block_tokenizer)):
        global_step += 1
        # with torch.no_grad():
        #     evaluator.build_reference(model, std_condition_concept)
        data_block = block_tokenizer[batch_i]
        
        # --- Process the Data Block ---
        labels_tensor = data_block['labels']
        tokenized_sentence1s = data_block['tokenized_sentence1s']
        tokenized_sentence2s = data_block['tokenized_sentence2s']
        
        model = model.train() # Set model to training mode
        sentence_pairs = [tokenized_sentence1s, tokenized_sentence2s]

        # Forward pass
        optimizer.zero_grad()
        with autocast(device_type, enabled=fp16):
            loss_value = train_loss(sentence_pairs, labels_tensor)

        # Backward pass
        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() # Step scheduler after optimizer step
        
        epoch_total_loss += loss_value.item()
        epoch_avg_loss = epoch_total_loss / (batch_i + 1)

        
        info = {
            "Global Step": global_step,
            "Epoch": epoch_i,
            "Batch": batch_i,
            "Loss": f"{loss_value.item():.4f}",
            "Avg Loss": f"{epoch_avg_loss:.4f}"
        }
        progress_bar.set_postfix(info)
        progress_bar.update(1)
        
        if global_step % wandb_report_steps == 0:
            wandb.log(info, step=global_step)
        
        ## save model every arg_saving_steps
        if global_step % arg_saving_steps == 0:
            print("Saving checkpoint...")
            auto_save_model(model, tokenizer, output_dir, max_saves=max_saves)
        
        #Log training loss periodically
        if False and global_step % arg_eval_steps == 0:
            model.eval()
            with torch.no_grad():
                eval_metrics = evaluator(query, n_results=100, k_list= [1, 10, 50])
                eval_results = {f"eval/{k}": v for k, v in eval_metrics.items()}
                wandb.log(eval_results, step=global_step)
            print(f"Evaluation results: {eval_metrics}")
            
            eval_accuracy = eval_metrics['top 1']
            # 4. Save Best Model
            if eval_accuracy > best_eval_accuracy:
                print(f"New best model found! accuracy improved from {best_eval_accuracy:.4f} to {eval_accuracy:.4f}. Saving...")
                best_eval_accuracy = eval_accuracy
                save_best_model(model, tokenizer, output_dir) # Pass necessary args
                wandb.log({"eval/best_accuracy": best_eval_accuracy}, step=global_step)
            



# --- End of Training ---
progress_bar.close()
wandb.finish()
logger.done()





