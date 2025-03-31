## Find the best batch size for training

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch
import pandas as pd
from datetime import datetime
# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset
from modules.ML_data import get_matching, get_relation, get_matching_validation,get_relation_positive_validation, DictBatchSampler
from modules.transformer import get_base_model


## make sure cuda is available
torch.cuda.is_available()


base_model = 'models/all-MiniLM-L6-v2'
# base_model = 'models/base_exclude_CIEL/checkpoint-65000'

special_tokens = ['[MATCHING]', '[OFFSPRINT]', '[ANCESTOR]']
## what datasets to use?
ds_names = ['matching', 'offspring', 'ancestor']

## columns to use
cols = ['sentence1', 'sentence2', 'label']

n_neg_matching = 4
n_neg_relation = 4
dt_seed = 42
data_folder = 'data/ML'
iterable_matching = get_matching(data_folder, n_neg=n_neg_matching, seed=dt_seed)
iterable_offspring, iterable_ancestor = get_relation(data_folder, n_neg = n_neg_relation, seed=dt_seed)
iterable_matching_validation = get_matching_validation(data_folder, seed=dt_seed)
offspring_validation,ancestor_validation = get_relation_positive_validation(data_folder)



model, tokenizer = get_base_model(base_model, special_tokens)

train_loss = losses.ContrastiveLoss(model=model)

arg_batch_size = 256
arg_eval_steps = 1024
arg_saving_steps = 1024*2


train_dataset = {
    'matching': iterable_matching,
    'offspring': iterable_offspring,
    'ancestor': iterable_ancestor
}
train_dataset = {k: train_dataset[k] for k in ds_names}

print(train_dataset)

## TODO: validate the sampling ratio gives the expected number of samples
block_size = 64*1024
shuffle_buffer = 64*1024
ratios = {'matching': 1, 'offspring': 0.5, 'ancestor': 0.5}
ratios = {k: ratios[k] for k in ds_names}
sampler = DictBatchSampler(train_dataset, batch_size=block_size, ratios = ratios, shuffle_buffer=shuffle_buffer)
ds_train = next(iter(sampler))


batch_sizes = [192, 256, 320, 512]
times = []

for i in range(len(batch_sizes)):
    arg_batch_size = batch_sizes[i]
    args = SentenceTransformerTrainingArguments(
        output_dir="test",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=arg_batch_size,
        per_device_eval_batch_size=arg_batch_size,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if your GPU can't handle FP16
        bf16=False,  # Set to True if your GPU supports BF16
        # batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates
        # Optional tracking/debugging parameters:
        save_strategy="no",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=ds_train,
        loss=train_loss,
        args=args,
    )

    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    times.append((arg_batch_size, elapsed_time))
    print(f"Batch size: {arg_batch_size}, Time taken: {elapsed_time} seconds")
    
    
