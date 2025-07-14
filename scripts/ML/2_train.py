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

from modules.ModelFunctions import auto_save_model, save_best_model, get_loss, get_base_model, get_ST_model
from modules.BlockTokenizer import BlockTokenizer
from modules.timed_logger import logger
from modules.Dataset import PositiveDataset, NegativeDataset, FalsePositiveDataset, CombinedDataset
from modules.metrics import evaluate_embedding_similarity_with_mrr
logger.reset_timer()



# --- Initialization ---
logger.log("Loading Model")
output_dir = f"output/finetune/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
base_model = 'ClinicalBERT'

model, tokenizer = get_ST_model(base_model)
#################################
## Load training data
#################################
logger.log("Loading training data")

base_path = "data/matching"
seed = 42
n_pos_matching = 10
n_neg_matching = 50
n_neg_relation = 50
n_fp_matching = 50

target_concepts = pd.read_feather(os.path.join(base_path, 'std_condition_concept.feather'))
name_bridge = pd.read_feather(os.path.join(base_path, 'condition_matching_name_bridge_train.feather'))
name_table = pd.read_feather(os.path.join(base_path, 'condition_matching_name_table_train.feather'))


#################################
## Creating datasets
#################################
matching_pos = PositiveDataset(
    target_concepts=target_concepts,
    name_table=name_table,
    name_bridge=name_bridge,
    max_elements=n_pos_matching,
    seed=seed
)

matching_neg = NegativeDataset(
    target_concepts=target_concepts,
    name_table=name_table,
    blacklist_bridge=name_bridge,
    max_elements = n_neg_matching,
    seed=seed
)



fp_path = os.path.join(base_path, f'fp_matching_{n_fp_matching}.feather')
matching_fp = FalsePositiveDataset(
    target_concepts=target_concepts,
    n_fp_matching=n_fp_matching,
    existing_path=fp_path
)
matching_fp.add_model(model)


ds_all = CombinedDataset(
    positive= matching_pos,
    negative= matching_neg,
    false_positive=matching_fp
)


# for i in tqdm(range(len(ds_all))):
#     if i == 8668897:
#         x = ds_all[i]  
#     pass
# len(ds_all)
# 8668897/10271679
# matching_fp.resample(model)

# ds_all=ds_all.shuffle(seed=seed)

#################################
## For validation
#################################
logger.log("Loading validation data")
condition_matching_valid = pd.read_feather(os.path.join(base_path, 'condition_matching_valid.feather'))


#################################
## Model
#################################
logger.log("Model settings")
fp16 = True # Use Mixed Precision (Float16)
learning_rate = 2e-5 # Typical default for AdamW with transformers
max_length = 512 # Maximum length for tokenization
# evaluator = CustomEvaluator()
max_saves = 4

# --- Initialization ---
wandb_report_steps = 256
temp_dir = tempfile.gettempdir()
wandb.init(project="Concept_Mapping", name=output_dir, dir = temp_dir)

# Loss and Optimizer
loss_func = losses.ContrastiveLoss(model=model)
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 4. Device Handling
device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
model = model.to(device)
loss_func = loss_func.to(device)
print(f"Using device: {device}")


# 7. Scheduler
epoch_num = 1
buffer_size = 256 * 16
batch_size = 256
arg_eval_steps = 512
arg_saving_steps = arg_eval_steps*2


block_tokenizer = BlockTokenizer(
    dataset = ds_all, 
    buffer_size = buffer_size, 
    batch_size= batch_size, 
    tokenizer = tokenizer, 
    device = device,
    max_length=max_length)

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


progress_bar = tqdm(total=len(block_tokenizer) * epoch_num)
for epoch_i in range(epoch_num):
    # evaluator.build_reference(model, std_condition_concept)  
    epoch_total_loss = 0.0
    
    if epoch_i > 0:
        ds_all = ds_all.resample(seed=epoch_i)
        ds_all = ds_all.shuffle(seed=epoch_i)
    
    for batch_i in range(len(block_tokenizer)):
        global_step += 1
        
        # Forward pass
        model = model.train() # Set model to training mode
        optimizer.zero_grad()
        with autocast(device_type, enabled=fp16):
            loss_value = get_loss(loss_func, block_tokenizer, batch_i)

        # Backward pass
        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() # Step scheduler after optimizer step
        
        epoch_total_loss += loss_value.item()
        epoch_avg_loss = epoch_total_loss / (batch_i + 1)

        info = {
            "total": global_step,
            "Loss": loss_value.item(),
            "Avg Loss": epoch_avg_loss,
            "Epoch": epoch_i,
            "Batch": batch_i
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
        if global_step % arg_eval_steps == 0:
            model.eval()
            with torch.no_grad():
                eval_results = evaluate_embedding_similarity_with_mrr(model, condition_matching_valid)
                ## move everything to evaluation panel
                eval_results = {f"eval/{k}": v for k, v in eval_results.items()}
                wandb.log(eval_results, step=global_step)
            print(f"Evaluation results: {eval_results}")

            eval_accuracy = eval_results["eval/MRR"]
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





