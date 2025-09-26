import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Fix tokenizer concurrency issues

import numpy as np
import pandas as pd
import wandb
import tempfile
import torch
import argparse
from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from datetime import datetime
from transformers import get_linear_schedule_with_warmup 
from sentence_transformers import losses, SentenceTransformer

from modules.ModelFunctions import auto_save_model, save_best_model, get_loss, get_ST_model, auto_load_model
from modules.BlockTokenizer import BlockTokenizer
from modules.timed_logger import logger
from modules.Dataset import PositiveDataset, NegativeDataset, CombinedDataset
from modules.metrics import evaluate_embedding_similarity_with_mrr
from modules.STHardNegMiner import mine_negatives

logger.reset_timer()


# --- CLI Args ---
parser = argparse.ArgumentParser(description="Train with positive-aware hard negative mining (built-in mine_hard_negatives) using all-MiniLM-L6-v2")
parser.add_argument("--no-relation", action="store_true", help="Disable relation data even if files exist or config says True")
parser.add_argument("--range-min", type=int, default=10, help="Minimum rank for candidate negatives")
parser.add_argument("--range-max", type=int, default=50, help="Maximum rank for candidate negatives")
parser.add_argument("--margin", type=float, default=0.05, help="Absolute margin for mining")
parser.add_argument("--num-neg-matching", type=int, default=None, help="Negatives per anchor for matching mining (override)")
parser.add_argument("--num-neg-relation", type=int, default=None, help="Negatives per anchor for relation mining (override)")
parser.add_argument("--sampling-strategy", type=str, choices=["random","top"], default="random", help="Negative sampling strategy from candidates")
parser.add_argument("--mine-batch-size", type=int, default=256, help="Batch size for miner embedding")
parser.add_argument("--no-faiss", action="store_true", help="Disable FAISS acceleration in miner")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
parser.add_argument("--buffer-mult", type=int, default=16, help="Buffer multiple of batch size for BlockTokenizer")
parser.add_argument("--model-checkpoint", type=str, default="none", help="Path to model checkpoint to load (default: none - use HF base model)")
parser.add_argument("--max-training-samples", type=int, default=None, help="Limit total training samples (default: None - use all data)")
parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
args = parser.parse_args()

# --- Initialization ---
logger.log("Loading Model (pos-aware training)")
output_dir = f"output/finetune_posaware/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
base_model = 'sentence-transformers/all-MiniLM-L6-v2'

# Load model - use Hugging Face model directly or specified checkpoint
if args.model_checkpoint and args.model_checkpoint != "none":
    model, tokenizer, train_config = auto_load_model(args.model_checkpoint)
else:
    model, tokenizer, train_config = None, None, {}

if model is None:
    # Load base ST model with special tokens for relation support
    logger.log(f"Loading base model with special tokens: {base_model}")
    model, tokenizer = get_ST_model(base_model)
    train_config = {}

start_epoch = train_config.get('epoch', 0)
start_batch_i = train_config.get('batch_i', 0) + 1 if 'batch_i' in train_config else 0
start_global_step = train_config.get('global_step', 0) + 1 if 'global_step' in train_config else 0
use_relation = train_config.get('use_relation', True)
if args.no_relation:
    use_relation = False


#################################
# Load training data
#################################
logger.log("Loading training data")

matching_base_path = "data/matching"
relation_base_path = "data/relation"
seed = 42

# Much smaller default dataset sizes
if not use_relation:
    n_pos_matching = 5  
    n_neg_matching = 10  
    n_fp_matching = 5   
else:
    n_pos_matching = 5  
    n_neg_matching = 5  
    n_fp_matching = 5   
    n_pos_relation = 5  
    n_neg_relation = 5  
    n_fp_relation = 5   

# Override mined negatives via CLI if provided
if args.num_neg_matching is not None:
    n_fp_matching = args.num_neg_matching
if use_relation and args.num_neg_relation is not None:
    n_fp_relation = args.num_neg_relation

target_concepts_path = os.path.join(matching_base_path, 'std_condition_concept.feather')
if not os.path.exists(target_concepts_path):
    # Fallback for older naming (if present)
    alt_path = os.path.join(matching_base_path, 'target_concepts.feather')
    target_concepts_path = alt_path if os.path.exists(alt_path) else target_concepts_path
target_concepts = pd.read_feather(target_concepts_path)
matching_name_bridge = pd.read_feather(os.path.join(matching_base_path, 'condition_matching_name_bridge_train.feather'))
matching_name_table = pd.read_feather(os.path.join(matching_base_path, 'condition_matching_name_table_train.feather'))


#################################
# Creating datasets
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
    max_elements=n_neg_matching,
    seed=seed
)

# Build anchor/positive pairs for matching and mine negatives with ST builtin
anchor_positive_match = matching_name_bridge
anchor_positive_match = anchor_positive_match.merge(
    matching_name_table[['name_id','name']].rename(columns={'name':'anchor'}),
    on='name_id'
).merge(
    target_concepts[['concept_id','concept_name']].rename(columns={'concept_name':'positive'}),
    on='concept_id'
)[['anchor','positive']].drop_duplicates()

# Limit training samples if specified
if args.max_training_samples is not None:
    logger.log(f"Limiting training data to {args.max_training_samples} samples")
    anchor_positive_match = anchor_positive_match.sample(n=min(args.max_training_samples, len(anchor_positive_match)), random_state=seed).reset_index(drop=True)

matching_fp_df = mine_negatives(
    anchor_positive_pairs=anchor_positive_match,
    model=model,
    range_min=args.range_min,
    range_max=args.range_max,
    margin=args.margin,
    num_negatives=n_fp_matching,
    sampling_strategy=args.sampling_strategy,
    batch_size=args.mine_batch_size,
    use_faiss=not args.no_faiss,
    verbose=True,
)


relation_table_path = os.path.join(relation_base_path, 'name_table_relation.feather')
relation_bridge_path = os.path.join(relation_base_path, 'name_bridge_relation.feather')
relation_available = os.path.exists(relation_table_path) and os.path.exists(relation_bridge_path)
if use_relation and relation_available:
    name_table_relation = pd.read_feather(relation_table_path)
    name_bridge_relation = pd.read_feather(relation_bridge_path)

    from modules.Dataset import PositiveDataset as RelPositiveDataset
    from modules.Dataset import NegativeDataset as RelNegativeDataset

    relation_pos = RelPositiveDataset(
        target_concepts=target_concepts,
        name_table=name_table_relation,
        name_bridge=name_bridge_relation,
        max_elements=n_pos_relation,
        seed=seed
    )

    relation_neg = RelNegativeDataset(
        target_concepts=target_concepts,
        name_table=name_table_relation,
        blacklist_bridge=name_bridge_relation,
        max_elements=n_neg_relation,
        seed=seed
    )

    anchor_positive_rel = name_bridge_relation
    anchor_positive_rel = anchor_positive_rel.merge(
        name_table_relation[['name_id','name']].rename(columns={'name':'anchor'}),
        on='name_id'
    ).merge(
        target_concepts[['concept_id','concept_name']].rename(columns={'concept_name':'positive'}),
        on='concept_id'
    )[['anchor','positive']].drop_duplicates()

    # Limit relation training samples if specified
    if args.max_training_samples is not None:
        relation_samples = min(args.max_training_samples // 2, len(anchor_positive_rel))  # Split between matching and relation
        anchor_positive_rel = anchor_positive_rel.sample(n=relation_samples, random_state=seed).reset_index(drop=True)

    relation_fp_df = mine_negatives(
        anchor_positive_pairs=anchor_positive_rel,
        model=model,
        range_min=args.range_min,
        range_max=args.range_max,
        margin=args.margin,
        num_negatives=n_fp_relation,
        sampling_strategy=args.sampling_strategy,
        batch_size=args.mine_batch_size,
        use_faiss=not args.no_faiss,
        verbose=True,
    )

if use_relation and relation_available:
    ds_all = CombinedDataset(
        matching_pos=matching_pos,
        matching_neg=matching_neg,
        matching_fp=matching_fp_df,   # pandas DataFrame accepted
        relation_pos=relation_pos,
        relation_fp=relation_fp_df,   # pandas DataFrame accepted
        relation_neg=relation_neg,
    )
else:
    ds_all = CombinedDataset(
        matching_pos=matching_pos,
        matching_neg=matching_neg,
        matching_fp=matching_fp_df,   # pandas DataFrame accepted
    )


#################################
# Validation data (OMOP CIEL)
#################################
logger.log("Loading validation data")
condition_matching_valid = None

# Try to load OMOP CIEL validation data first
try:
    from modules.ChromaVecDB import ChromaVecDB

    logger.log("Attempting to load OMOP CIEL validation data")
    conceptEX = pd.read_feather('data/omop_feather/conceptEX.feather')
    conditions = conceptEX[conceptEX['domain_id'] == 'Condition']
    std_conditions = conditions[conditions['standard_concept'] == 'S']
    nonstd_conditions = conditions[conditions['standard_concept'] != 'S']

    # Database: standard conditions
    database = std_conditions[['concept_id', 'concept_name']]

    # Query: CIEL concepts
    ciel_query_df = nonstd_conditions[
        ['concept_id', 'concept_name', 'std_concept_id']
    ][nonstd_conditions['vocabulary_id'] == 'CIEL']

    # Store CIEL validation data for later evaluation use
    global ciel_validation_data
    ciel_validation_data = {
        'database': database,
        'query_df': ciel_query_df
    }

    # Create a simple validation dataset for backward compatibility with existing evaluation code
    # This maintains the same format for the training loop evaluation
    y_true = ciel_query_df['std_concept_id'].str[0].astype(int)
    std_concept_names = std_conditions[['concept_id', 'concept_name']].rename(
        columns={'concept_id': 'std_concept_id', 'concept_name': 'std_concept_name'}
    )
    ciel_validation = ciel_query_df.merge(std_concept_names, left_on=y_true, right_on='std_concept_id', how='inner')

    condition_matching_valid = ciel_validation[['concept_name', 'std_concept_name']].rename(
        columns={'concept_name': 'query', 'std_concept_name': 'target'}
    )
    logger.log(f"Loaded OMOP CIEL validation data: {len(condition_matching_valid)} samples")
except Exception as e:
    logger.log(f"Could not load OMOP CIEL validation data: {e}")
    ciel_validation_data = None
    # Fallback to original validation file if available
    fallback_valid_path = os.path.join(matching_base_path, 'condition_matching_valid.feather')
    if os.path.exists(fallback_valid_path):
        try:
            condition_matching_valid = pd.read_feather(fallback_valid_path)
            logger.log(f"Using fallback validation data: {len(condition_matching_valid)} samples")
        except Exception as e2:
            logger.log(f"Could not load fallback validation data: {e2}")
            condition_matching_valid = None

# Load training subsets for evaluation
condition_matching_train_subset = None
train_subset_path = os.path.join(matching_base_path, 'condition_matching_train_subset.feather')
if os.path.exists(train_subset_path):
    try:
        condition_matching_train_subset = pd.read_feather(train_subset_path)
        logger.log(f"Loaded training subset for evaluation: {len(condition_matching_train_subset)} samples")
    except Exception as e:
        logger.log(f"Could not load training subset: {e}")

condition_relation_train_subset = None
rel_subset_path = os.path.join(relation_base_path, 'condition_relation_train_subset.feather')
if use_relation and relation_available and os.path.exists(rel_subset_path):
    try:
        condition_relation_train_subset = pd.read_feather(rel_subset_path)
        logger.log(f"Loaded relation subset for evaluation: {len(condition_relation_train_subset)} samples")
    except Exception as e:
        logger.log(f"Could not load relation subset: {e}")


#################################
# Model settings
#################################
logger.log("Model settings")
fp16 = True
learning_rate = 2e-5
max_length = 512
max_saves = 4

loss_func = losses.ContrastiveLoss(model=model)
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Device detection - CUDA only
if torch.cuda.is_available():
    device_type = "cuda"
    device = torch.device("cuda")
    print("Using CUDA GPU acceleration")
else:
    device_type = "cpu"
    device = torch.device("cpu")
    print("Using CPU")


model = model.to(device)
loss_func = loss_func.to(device)
print(f"Using device: {device}")

epoch_num = args.epochs
batch_size = args.batch_size
buffer_size = batch_size * args.buffer_mult
arg_eval_steps = 2048
arg_saving_steps = arg_eval_steps * 2

block_tokenizer = BlockTokenizer(
    dataset=ds_all,
    buffer_size=buffer_size,
    batch_size=batch_size,
    tokenizer=tokenizer,
    device=device,
    max_length=max_length,
)

warmup_ratio = 0.1
total_steps = len(ds_all) // batch_size * epoch_num
warmup_steps = int(total_steps * warmup_ratio)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(0, warmup_steps - start_global_step),
    num_training_steps=max(1, total_steps - start_global_step),
)


scaler = GradScaler(device=device_type, enabled=fp16)

global_step = start_global_step
best_eval_accuracy = float('-inf')

wandb_report_steps = 256
temp_dir = tempfile.gettempdir()
if not args.no_wandb:
    wandb.init(project="Concept_Mapping", name=output_dir, dir=temp_dir)
else:
    print("W&B logging disabled")

progress_bar = tqdm(total=len(block_tokenizer) * epoch_num)
progress_bar.update(start_global_step)

for epoch_i in range(start_epoch, epoch_num):
    epoch_total_loss = 0.0

    if epoch_i > 0:
        model.eval()
        with torch.no_grad():
            # Resample negatives each epoch to refresh hard negatives
            if hasattr(ds_all, 'resample'):
                ds_all = ds_all.resample(seed=epoch_i)
            ds_all = ds_all.shuffle(seed=epoch_i)
            block_tokenizer.update_dataset(ds_all)

    batch_start = start_batch_i if epoch_i == start_epoch else 0
    for batch_i in range(batch_start, len(block_tokenizer)):
        global_step += 1

        cur_train_config = {
            'global_step': global_step,
            'epoch': epoch_i,
            'batch_i': batch_i,
            'use_relation': use_relation,
            'posaware': True,
        }

        model.train()
        optimizer.zero_grad()
        with autocast(device_type, enabled=fp16):
            loss_value = get_loss(loss_func, block_tokenizer, batch_i)

        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_total_loss += loss_value.item()
        epoch_avg_loss = epoch_total_loss / (batch_i - batch_start + 1)

        info = {
            "total": global_step,
            "Loss": loss_value.item(),
            "Avg Loss": epoch_avg_loss,
            "Epoch": epoch_i,
            "Batch": batch_i,
        }
        progress_bar.set_postfix(info)
        progress_bar.update(1)

        if global_step % wandb_report_steps == 0 and not args.no_wandb:
            wandb.log(info, step=global_step)

        if global_step % arg_saving_steps == 0:
            print("Saving checkpoint...")
            auto_save_model(model, tokenizer, output_dir, max_saves=max_saves, train_config=cur_train_config)

        if global_step % arg_eval_steps == 0:
            model.eval()
            with torch.no_grad():
                eval_results = {}
                if condition_matching_valid is not None:
                    try:
                        # Set single thread for evaluation to avoid tokenizer conflicts
                        import os
                        os.environ["TOKENIZERS_PARALLELISM"] = "false"
                        eval_metrics = evaluate_embedding_similarity_with_mrr(model, condition_matching_valid)
                        eval_results = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    except Exception as e:
                        logger.log(f"Evaluation failed: {e}")
                        eval_results["eval/MRR"] = 0.0  # Default value for best model tracking

                if condition_matching_train_subset is not None:
                    try:
                        train_matching_results = evaluate_embedding_similarity_with_mrr(model, condition_matching_train_subset)
                        train_matching_results = {f"train_matching/{k}": v for k, v in train_matching_results.items()}
                        eval_results.update(train_matching_results)
                    except Exception as e:
                        logger.log(f"Train matching evaluation failed: {e}")

                if use_relation and condition_relation_train_subset is not None:
                    try:
                        train_relation_results = evaluate_embedding_similarity_with_mrr(model, condition_relation_train_subset)
                        train_relation_results = {f"train_relation/{k}": v for k, v in train_relation_results.items()}
                        eval_results.update(train_relation_results)
                    except Exception as e:
                        logger.log(f"Train relation evaluation failed: {e}")

                if not args.no_wandb:
                    wandb.log(eval_results, step=global_step)
            print(f"Evaluation results: {eval_results}")

            eval_accuracy = eval_results.get("eval/MRR", 0.0)
            if eval_accuracy > best_eval_accuracy:
                print(f"New best model found! accuracy improved from {best_eval_accuracy:.4f} to {eval_accuracy:.4f}. Saving...")
                best_eval_accuracy = eval_accuracy
                save_best_model(model, tokenizer, output_dir, train_config=cur_train_config)
                if not args.no_wandb:
                    wandb.log({"eval/best_accuracy": best_eval_accuracy}, step=global_step)

progress_bar.close()

# Final CIEL evaluation using ChromaVecDB approach (like poster.py)
if 'ciel_validation_data' in globals() and ciel_validation_data is not None:
    try:
        logger.log("Performing final CIEL evaluation with ChromaVecDB")

        database = ciel_validation_data['database']
        query_df = ciel_validation_data['query_df']

        # Create vector database and query
        db = ChromaVecDB(model=model, name='final_ciel_eval', path=None)
        db.empty_collection()
        db.store_concepts(database, batch_size=5461)

        max_k = 50
        res = db.query(query_df[['concept_name']], n_results=max_k)

        y_true = query_df['std_concept_id'].str[0].astype(int).tolist()
        pred_lists = res['ids']

        # Calculate comprehensive metrics
        k_list = [1, 10, 50]
        final_ciel_metrics = {}

        for k in k_list:
            hits = [1 if (y_true[i] in pred_lists[i][:k]) else 0 for i in range(len(y_true))]
            acc_k = sum(hits) / len(hits)
            final_ciel_metrics[f'Final_Accuracy@{k}'] = acc_k

        # Calculate MRR
        reciprocal_ranks = []
        for true_id, preds in zip(y_true, pred_lists):
            try:
                rank = preds.index(true_id) + 1
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        final_ciel_metrics['Final_MRR'] = mrr

        print("\n=== Final CIEL Evaluation Results ===")
        print(f"Final MRR: {mrr:.4f}")
        print(f"Final Accuracy@1: {final_ciel_metrics['Final_Accuracy@1']:.4f}")
        print(f"Final Accuracy@10: {final_ciel_metrics['Final_Accuracy@10']:.4f}")
        print(f"Final Accuracy@50: {final_ciel_metrics['Final_Accuracy@50']:.4f}")

        if not args.no_wandb:
            wandb.log(final_ciel_metrics)

        # Save final metrics to file
        try:
            import json
            with open(os.path.join(output_dir, 'final_ciel_metrics.json'), 'w') as f:
                json.dump(final_ciel_metrics, f, indent=2)
        except Exception as e:
            logger.log(f"Warning: Could not save final CIEL metrics: {e}")

    except Exception as e:
        logger.log(f"Final CIEL evaluation failed: {e}")

if not args.no_wandb:
    wandb.finish()
logger.done()
