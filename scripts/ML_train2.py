import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset


## make sure cuda is available
torch.cuda.is_available()

## Load the dataset
positive_sample_dataset = pd.read_feather('data/ML/conceptML_positive.feather')
negative_sample_dataset = pd.read_feather('data/ML/conceptML_negative.feather')
dataset_pd = pd.concat([positive_sample_dataset, negative_sample_dataset])

## Check for missing values
dataset_pd[dataset_pd[['sentence1','sentence2','label1']].isnull().any(axis=1)]


## pandas dataframe to datasets
dataset_pd = dataset_pd.reset_index(drop=True)
dataset = Dataset.from_pandas(dataset_pd)

## rename label1 to label
dataset = dataset.rename_column('label1', 'label')
## keep only sentence1, sentence2, and label
dataset = dataset.remove_columns(['concept_id1', 'concept_id2', 'source', 'index'])

subset = dataset.select(range(1024*8))

np.sum(subset['label'])

## trainint and testing split
dt = subset.train_test_split(test_size=0.005, seed=42)
train_dataset = dt['train']
test_dataset = dt['test']


model = SentenceTransformer('models/all-MiniLM-L6-v2')
train_loss = losses.ContrastiveLoss(model=model)

output_dir = "models/fine-tuned2"

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=10,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if your GPU can't handle FP16
    bf16=False,  # Set to True if your GPU supports BF16
    # batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    run_name="mpnet-base-all-nli-triplet",  # Used in W&B if `wandb` is installed
)

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=train_loss,
    args=args,
)
trainer.train()

## save the model
model.save(f"{output_dir}/final")