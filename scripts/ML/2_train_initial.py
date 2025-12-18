import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import numpy as np
import pandas as pd
import wandb
import torch
import argparse
from tqdm import tqdm
from torch.optim.adamw import AdamW
from datetime import datetime
from sentence_transformers import losses, SentenceTransformer
from types import SimpleNamespace

from modules.ModelFunctions import get_ST_model, auto_load_model
from modules.timed_logger import logger
from modules.metrics import evaluate_embedding_similarity_with_mrr
from modules.STHardNegMiner import mine_negatives, _pairs_to_dataset
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers import InputExample
from datasets import Dataset


def to_hf_dataset(dataset_iterable):
    rows = []
    for item in tqdm(dataset_iterable):
        rows.append({
            "sentence1": item["sentence1"],
            "sentence2": item["sentence2"],
            "label": int(item["label"])   # ensure 0/1
        })
    ds = Dataset.from_list(rows)
    # Ensure correct column order for the loss: [inputs..., label]
    ds = ds.select_columns(["sentence1", "sentence2", "label"])
    return ds



logger.reset_timer()






args = {
    "test_mode": False,
    
    "no_relation": False,  # Disable relation data even if files exist or config says True
    "range_min": 10,       # Minimum rank for candidate negatives
    "range_max": 50,       # Maximum rank for candidate negatives
    "relative_margin": 0.01,  # Relative margin for mining
    "num_neg_matching": None,  # Negatives per anchor for matching mining (override)
    "num_neg_relation": None,  # Negatives per anchor for relation mining (override)
    "sampling_strategy": "top",  # Negative sampling strategy from candidates
    "batch_size": 256,      # Batch size for training
    "no_faiss": False,           # Disable FAISS acceleration in miner
    "model_checkpoint": "none",  # Path to model checkpoint to load (default: none - use HF base model)
    
    
    "use_relation": True,
    "n_pos_matching": 5,
    "n_neg_matching": 5,
    "n_fp_matching": 5,
    "n_pos_relation": 5,
    "n_neg_relation": 5,
    "n_fp_relation": 5,
    
}

args = SimpleNamespace(**args)
