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

model, tokenizer = auto_load_model("output/all-MiniLM-L6-v2_2025-04-17_20-01-05")


model2, tokenizer2 = auto_load_model("output/all-MiniLM-L6-v2_2025-04-17_20-01-05")



for i in range(11,10):
    print(i)
    
    

evaluate_embedding_similarity_with_mrr(model, condition_matching_valid, threshold=0.8)