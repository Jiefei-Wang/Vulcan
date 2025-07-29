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

from modules.ModelFunctions import auto_save_model, save_best_model, get_loss, get_base_model, get_ST_model, auto_load_model
from modules.BlockTokenizer import BlockTokenizer
from modules.timed_logger import logger
from modules.Dataset import PositiveDataset, NegativeDataset, FalsePositiveDataset, CombinedDataset
from modules.metrics import evaluate_embedding_similarity_with_mrr

from modules.FaissDB import is_initialized
logger.reset_timer()



# --- Initialization ---
logger.log("Loading Model")
output_dir = f"output/finetune/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
base_model = 'ClinicalBERT'

## selection between original model and trained model

use_relation = True


if True:
    if use_relation:
        # continue training with relation data
        model, tokenizer, _ = auto_load_model("output/relation_with_token")
    else:
        # start from scratch
        model, tokenizer = get_ST_model(base_model)
    start_epoch = 0
    start_batch_i = 0
    start_global_step = 0
else:
    model, tokenizer, train_config = auto_load_model("output/finetune/2025-07-16_17-39-33")

    start_epoch = train_config.get('epoch', 0)
    start_batch_i = train_config.get('batch_i', 0) + 1
    start_global_step = train_config.get('global_step', 0) + 1
    use_relation = train_config.get('use_relation', False)
    

#################################
## Load training data
#################################
logger.log("Loading training data")

matching_base_path = "data/matching"
relation_base_path = "data/relation"
seed = 42

if not use_relation:
    n_pos_matching = 20
    n_neg_matching = 50
    n_fp_matching = 50
else:
    n_pos_matching = 20
    n_neg_matching = 20
    n_fp_matching = 20
    n_pos_relation = 20
    n_neg_relation = 20
    n_fp_relation = 20
    



target_concepts = pd.read_feather(os.path.join(matching_base_path, 'target_concepts.feather'))
matching_name_bridge = pd.read_feather(os.path.join(matching_base_path, 'condition_matching_name_bridge_train.feather'))
matching_name_table = pd.read_feather(os.path.join(matching_base_path, 'condition_matching_name_table_train.feather'))


#################################
## Creating datasets
#################################
matching_pos = PositiveDataset(
    target_concepts=target_concepts,
    name_table=matching_name_table,
    name_bridge=matching_name_bridge,
    max_elements=n_pos_matching,
    seed=seed
)

matching_neg = NegativeDataset(
    target_concepts=target_concepts,
    name_table=matching_name_table,
    blacklist_bridge=matching_name_bridge,
    max_elements = n_neg_matching,
    seed=seed
)



# matching_fp_path = os.path.join(matching_base_path, f'matching_fp_{n_fp_matching}.feather')
matching_fp = FalsePositiveDataset(
    corpus_ids=target_concepts['concept_id'],
    corpus_names=target_concepts['concept_name'],
    n_fp=n_fp_matching,
    # existing_path=matching_fp_path,
    repos='training_target_false_positive'
)
matching_fp.add_model(model)
matching_fp.resample()

is_initialized("training_target_false_positive")


if use_relation:
    name_table_relation = pd.read_feather(os.path.join(relation_base_path, 'name_table_relation.feather'))
    name_bridge_relation = pd.read_feather(os.path.join(relation_base_path, 'name_bridge_relation.feather'))
    
    relation_pos = PositiveDataset(
        target_concepts=target_concepts,
        name_table=name_table_relation,
        name_bridge=name_bridge_relation,
        max_elements=n_pos_relation,
        seed=seed
    )
    
    relation_neg = NegativeDataset(
        target_concepts=target_concepts,
        name_table=name_table_relation,
        blacklist_bridge=name_bridge_relation,
        max_elements=n_neg_relation,
        seed=seed
    )

    # relation_fp_path = os.path.join(relation_base_path, f'fp_relation_{n_fp_relation}.feather')
    relation_fp = FalsePositiveDataset(
        corpus_ids=target_concepts['concept_id'],
        corpus_names=target_concepts['concept_name'],
        query_ids=name_table_relation['name_id'],
        query_names=name_table_relation['name'],
        n_fp=n_fp_relation,
        blacklist_from=name_bridge_relation['concept_id'],
        blacklist_to=name_bridge_relation['name_id'],
        # existing_path=relation_fp_path,
        repos='training_target_false_positive'
    )
    relation_fp.add_model(model)
    relation_fp.resample()




if use_relation:
    ds_all = CombinedDataset(
        matching_pos= matching_pos,
        matching_neg= matching_neg,
        matching_fp=matching_fp,
        relation_pos=relation_pos,
        relation_fp=relation_fp,
        relation_neg=relation_neg
    )
else:
    ds_all = CombinedDataset(
        matching_pos= matching_pos,
        matching_neg= matching_neg,
        matching_fp=matching_fp
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
condition_matching_valid = pd.read_feather(os.path.join(matching_base_path, 'condition_matching_valid.feather'))

condition_matching_train_subset = pd.read_feather(os.path.join(matching_base_path, 'condition_matching_train_subset.feather'))

condition_relation_train_subset = pd.read_feather(os.path.join(relation_base_path, 'condition_relation_train_subset.feather'))

#################################
## Model
#################################
logger.log("Model settings")
fp16 = True # Use Mixed Precision (Float16)
learning_rate = 2e-5 # Typical default for AdamW with transformers
max_length = 512 # Maximum length for tokenization
# evaluator = CustomEvaluator()
max_saves = 4


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
epoch_num = 10
buffer_size = 256 * 16
batch_size = 256
arg_eval_steps = 2048
arg_saving_steps = arg_eval_steps*2


block_tokenizer = BlockTokenizer(
    dataset = ds_all, 
    buffer_size = buffer_size, 
    batch_size= batch_size, 
    tokenizer = tokenizer, 
    device = device,
    max_length=max_length)

# warmup period
warmup_ratio = 0.1
total_steps = len(ds_all) // batch_size * epoch_num
warmup_steps = int(total_steps * warmup_ratio)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(0, warmup_steps - start_global_step),
    num_training_steps=max(1, total_steps - start_global_step)
)

# 8. Mixed Precision Scaler
scaler = GradScaler(device=device_type, enabled=fp16)

# --- Manual Training Loop ---
global_step = start_global_step
best_eval_accuracy = float('-inf')




# --- log ---
wandb_report_steps = 256
temp_dir = tempfile.gettempdir()
wandb.init(project="Concept_Mapping", name=output_dir, dir = temp_dir)

progress_bar = tqdm(total=len(block_tokenizer) * epoch_num)
progress_bar.update(start_global_step)


for epoch_i in range(start_epoch, epoch_num):
    # evaluator.build_reference(model, std_condition_concept)  
    epoch_total_loss = 0.0
    
    if epoch_i > 0:
        model.eval()
        with torch.no_grad():
            ds_all = ds_all.resample(seed=epoch_i)
            ds_all = ds_all.shuffle(seed=epoch_i)
            
            block_tokenizer.update_dataset(ds_all)

    batch_start = start_batch_i if epoch_i == start_epoch else 0
    for batch_i in range(batch_start, len(block_tokenizer)):
        global_step += 1
        
        train_config = {
            'global_step': global_step,
            'epoch': epoch_i,
            'batch_i': batch_i
        }
        
        # Forward pass
        model.train() # Set model to training mode
        optimizer.zero_grad()
        with autocast(device_type, enabled=fp16):
            loss_value = get_loss(loss_func, block_tokenizer, batch_i)

        # Backward pass
        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() # Step scheduler after optimizer step
        
        epoch_total_loss += loss_value.item()
        epoch_avg_loss = epoch_total_loss / (batch_i - batch_start + 1)

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
            auto_save_model(model, tokenizer, output_dir, max_saves=max_saves, train_config=train_config)
        
        #Log training loss periodically
        if global_step % arg_eval_steps == 0:
            model.eval()
            with torch.no_grad():
                eval_results = evaluate_embedding_similarity_with_mrr(model, condition_matching_valid)
                ## move everything to evaluation panel
                eval_results = {f"eval/{k}": v for k, v in eval_results.items()}
                
                train_matching_results = evaluate_embedding_similarity_with_mrr(model, condition_matching_train_subset)
                train_matching_results = {f"train_matching/{k}": v for k, v in train_matching_results.items()}
                eval_results.update(train_matching_results)
                
                train_relation_results = evaluate_embedding_similarity_with_mrr(model, condition_relation_train_subset)
                train_relation_results = {f"train_relation/{k}": v for k, v in train_relation_results.items()}
                eval_results.update(train_relation_results)
                
                wandb.log(eval_results, step=global_step)
            print(f"Evaluation results: {eval_results}")

            eval_accuracy = eval_results["eval/MRR"]
            # 4. Save Best Model
            if eval_accuracy > best_eval_accuracy:
                print(f"New best model found! accuracy improved from {best_eval_accuracy:.4f} to {eval_accuracy:.4f}. Saving...")
                best_eval_accuracy = eval_accuracy
                save_best_model(model, tokenizer, output_dir, train_config=train_config) # Pass necessary args
                wandb.log({"eval/best_accuracy": best_eval_accuracy}, step=global_step)
        



# --- End of Training ---
progress_bar.close()
wandb.finish()
logger.done()





