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


model_name = 'models/all-MiniLM-L6-v2'
# model_name = 'models/base_exclude_CIEL/checkpoint-65000'
output_dir = "models/base_exclude_CIEL2"

ds_names = ['matching', 'relation']

conceptEX = pd.read_feather('data/omop_feather/conceptEX.feather')
matching_all = pd.read_feather('data/ML/conceptML_matching.feather')
relation_all = pd.read_feather('data/ML/conceptML_relation.feather')
matching_validation = pd.read_feather('data/ML/conceptML_matching_validation.feather')

matching_validation_sub = matching_validation.sample(frac=0.1, random_state=42)
dataset_matching = Dataset.from_pandas(matching_all)
dataset_relation = Dataset.from_pandas(relation_all)

print(dataset_matching)
print(dataset_relation)

## empty dataset
relation_validation = Dataset.from_dict(dataset_relation[:0])


## train test split
# dataset_matching_split = dataset_matching.train_test_split(test_size=0.0005, seed=42)
# dataset_relation_split = dataset_relation.train_test_split(test_size=0.0001, seed=42)

# dataset_matching_train = dataset_matching_split['train']
# dataset_matching_test = dataset_matching_split['test']
# dataset_relation_train = dataset_relation_split['train']
# dataset_relation_test = dataset_relation_split['test']


train_dataset = {
    'matching': dataset_matching,
    'relation': dataset_relation
}
train_dataset = {k: train_dataset[k] for k in ds_names}
test_dataset = {
    'matching': matching_validation_sub,
    'relation': relation_validation
}
test_dataset = {k: test_dataset[k] for k in ds_names}

print(train_dataset)
print(test_dataset)



model = SentenceTransformer(model_name)

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
    

from sentence_transformers.evaluation import BinaryClassificationEvaluator

matching_validation_sub.reset_format()
# Initialize the evaluator
dev_evaluator1 = BinaryClassificationEvaluator(
    sentences1=matching_validation_sub["sentence1"],
    sentences2=matching_validation_sub["sentence2"],
    labels=matching_validation_sub["label"],
    name="evaluation1",
)
dev_evaluator1(model)

train_loss = {
    'matching' : losses.ContrastiveLoss(model=model),
    'relation' : losses.ContrastiveLoss(model=model)
}
train_loss = {k: train_loss[k] for k in ds_names}

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=5,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if your GPU can't handle FP16
    bf16=False,  # Set to True if your GPU supports BF16
    # batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=10000,
    save_total_limit=2,
    logging_steps=500,
    run_name=output_dir,  # Used in W&B if `wandb` is installed
)

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=train_loss,
    args=args,
    evaluator=dev_evaluator1,
)


trainer.train()


model.save(f"{output_dir}/final")