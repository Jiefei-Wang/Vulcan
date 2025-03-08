import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch
import pandas as pd
import numpy as np
from datetime import datetime
# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, models
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset, DatasetDict, concatenate_datasets, IterableDataset
from modules.ML_data import get_matching, get_relation, get_matching_validation,get_relation_positive_validation
from sentence_transformers.evaluation import BinaryClassificationEvaluator



## make sure cuda is available
torch.cuda.is_available()


base_model = 'models/all-MiniLM-L6-v2'
# base_model = 'models/base_exclude_CIEL/checkpoint-65000'
output_dir = f"models/{base_model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

special_tokens = ['[MATCHING]', '[OFFSPRINT]', '[ANCESTOR]']
## what datasets to use?
ds_names = ['matching', 'offspring', 'ancestor']

n_neg_matching = 4
n_neg_relation = 4
dt_seed = 42
data_folder = 'data/ML'
iterable_matching = get_matching(data_folder, n_neg=n_neg_matching, seed=dt_seed)
iterable_offspring, iterable_ancestor = get_relation(data_folder, n_neg = n_neg_relation, seed=dt_seed)
iterable_matching_validation = get_matching_validation(data_folder, seed=dt_seed)
offspring_validation,ancestor_validation = get_relation_positive_validation(data_folder)


for example in iterable_matching.trainer_iter():
    print(example)
    break


## Turn them into huggingface datasets
## Training
cols = ['sentence1', 'sentence2', 'label']
seed_shuffle = 42
buffer_size = 1000
matching_ds = IterableDataset.from_generator(iterable_matching.trainer_iter).shuffle(seed=seed_shuffle, buffer_size=buffer_size)
offspring_ds = IterableDataset.from_generator(iterable_offspring.trainer_iter).shuffle(seed=seed_shuffle, buffer_size=buffer_size)
ancestor_ds = IterableDataset.from_generator(iterable_ancestor.trainer_iter).shuffle(seed=seed_shuffle, buffer_size=buffer_size)



## get the first valid_size rows for validation
valid_size = 1000
it = iterable_matching_validation.trainer_iter()
matching_validation_pd = pd.DataFrame([next(it) for i in range(valid_size)])

matching_validation_ds = Dataset.from_pandas(matching_validation_pd[cols])
offspring_validation_ds = Dataset.from_pandas(offspring_validation[cols].sample(valid_size, random_state=42))
ancestor_validation_ds = Dataset.from_pandas(ancestor_validation[cols].sample(valid_size, random_state=42))


for example in matching_ds:
    print(example)
    break


train_dataset = {
    'matching': matching_ds,
    'offspring': offspring_ds,
    'ancestor': ancestor_ds
}
validation_dataset = {
    'matching': matching_validation_ds,
    'offspring': offspring_validation_ds,
    'ancestor': ancestor_validation_ds
}

print(train_dataset)
print(validation_dataset)



model = SentenceTransformer(base_model)

## add special tokens
transformer = model[0]  
tokenizer = transformer.tokenizer
auto_model = transformer.auto_model

# Now you can add special tokens to the tokenizer
special_tokens_dict = {
    "additional_special_tokens": special_tokens
}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

# Resize the model embeddings if we added tokens
if num_added_tokens > 0:
    auto_model.resize_token_embeddings(len(tokenizer))
    

matching_validation_ds.reset_format()
# Initialize the evaluator
dev_evaluator1 = BinaryClassificationEvaluator(
    sentences1=matching_validation_ds["sentence1"],
    sentences2=matching_validation_ds["sentence2"],
    labels=matching_validation_ds["label"],
    name="matching_eval",
)
dev_evaluator1(model)

train_loss = {
    'matching' : losses.ContrastiveLoss(model=model),
    'offspring' : losses.ContrastiveLoss(model=model),
    'ancestor' : losses.ContrastiveLoss(model=model)
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
    eval_dataset=validation_dataset,
    loss=train_loss,
    args=args,
    evaluator=dev_evaluator1,
)


trainer.train()


model.save(f"{output_dir}/final")