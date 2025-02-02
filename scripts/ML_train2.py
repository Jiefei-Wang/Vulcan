import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch
import pandas as pd
import numpy as np
# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, models
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset, DatasetDict, concatenate_datasets

## make sure cuda is available
torch.cuda.is_available()

dataset_dict = DatasetDict.load_from_disk('data/train_dataset_dict')

## combine matching and relation
dataset_matching = dataset_dict['matching']
dataset_relation = dataset_dict['relation']

dataset = concatenate_datasets([dataset_matching, dataset_relation])

subset = dataset.select(range(1024*8))

np.sum(subset['label'])

## trainint and testing split
dt = subset.train_test_split(test_size=0.005, seed=42)
train_dataset = dt['train']
test_dataset = dt['test']


model = SentenceTransformer('models/all-MiniLM-L6-v2')

## add special tokens
transformer = model[0]  
tokenizer = transformer.tokenizer
auto_model = transformer.auto_model

# Now you can add special tokens to the tokenizer
special_tokens_dict = {
    "additional_special_tokens": ["[MATCHING]", "[RELATION]"]
}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

# Resize the model embeddings if we added tokens
if num_added_tokens > 0:
    auto_model.resize_token_embeddings(len(tokenizer))



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