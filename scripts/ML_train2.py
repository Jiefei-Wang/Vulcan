import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import math
import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import pandas as pd
from datetime import datetime
import wandb
import tempfile
import concurrent.futures
from transformers import get_linear_schedule_with_warmup 
from sentence_transformers import losses
from modules.ML_data import get_matching, get_relation, get_matching_validation,get_relation_positive_validation, DictBatchSampler, future_tokenize
from modules.ML_train import get_base_model, auto_save_model, save_best_model, CustomEvaluator
from modules.TOKENS import TOKENS
from tqdm.auto import tqdm 

import pandas as pd
import os
from modules.timed_logger import logger
from tqdm import tqdm
logger.reset_timer()
logger.log("")

base_path = "data/matching"
std_condition_concept = pd.read_feather(os.path.join(base_path, 'std_condition_concept.feather'))
condition_matching_name_bridge_train = pd.read_feather(os.path.join(base_path, 'condition_matching_name_bridge_train.feather'))
condition_matching_name_table_train = pd.read_feather(os.path.join(base_path, 'condition_matching_name_table_train.feather'))





base_model = 'all-MiniLM-L6-v2'
base_model_path = f'models/{base_model}'
logger.log("Loading model")
model, tokenizer = get_base_model(base_model_path, special_tokens)

evaluator = CustomEvaluator()


mini_batch_size = 256 
arg_eval_steps = 2048
arg_saving_steps = arg_eval_steps*2

fp16 = True # Use Mixed Precision (Float16)
warmup_ratio = 0.1
learning_rate = 2e-5 # Typical default for AdamW with transformers

max_saves = 4
epoch_num = 5

# --- Initialization ---
wandb_report_steps = 256
temp_dir = tempfile.gettempdir()
wandb.init(project="Concept_Mapping", name=output_dir, dir = temp_dir)

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










