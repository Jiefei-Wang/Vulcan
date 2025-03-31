import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch
import pandas as pd
import numpy as np
from datetime import datetime
from math import ceil
import logging
import wandb
# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset
from modules.ML_data import get_matching, get_relation, get_matching_validation,get_relation_positive_validation, DictBatchSampler
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from modules.ML_train import get_base_model, auto_save_model

## disable default huggingface logging
logging.getLogger("transformers").setLevel(logging.WARNING)  # Or logging.ERROR
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


## make sure cuda is available
torch.cuda.is_available()


base_model = 'all-MiniLM-L6-v2'
base_model_path = f'models/{base_model}'
# base_model = 'models/base_exclude_CIEL/checkpoint-65000'
output_dir = f"output/{base_model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

special_tokens = ['[MATCHING]', '[OFFSPRINT]', '[ANCESTOR]']
## what datasets to use?
ds_names = ['matching', 'offspring', 'ancestor']

## columns to use
cols = ['sentence1', 'sentence2', 'label']


n_neg_matching = 10
n_neg_relation = 10
dt_seed = 42
data_folder = 'data/ML'
iterable_matching = get_matching(data_folder, n_neg=n_neg_matching, seed=dt_seed)
iterable_offspring, iterable_ancestor = get_relation(data_folder, n_neg = n_neg_relation, seed=dt_seed)
iterable_matching_validation = get_matching_validation(data_folder, seed=dt_seed)
offspring_validation,ancestor_validation = get_relation_positive_validation(data_folder)


for example in iterable_matching.trainer_iter():
    print(example)
    break


## get the first valid_size rows for validation
valid_size = 1000
it = iterable_matching_validation.trainer_iter()
matching_validation_pd = pd.DataFrame([next(it) for i in range(valid_size)])

matching_validation_ds = Dataset.from_pandas(matching_validation_pd[cols])
offspring_validation_ds = Dataset.from_pandas(offspring_validation[cols].sample(valid_size, random_state=42))
ancestor_validation_ds = Dataset.from_pandas(ancestor_validation[cols].sample(valid_size, random_state=42))


model, tokenizer = get_base_model(base_model_path, special_tokens)


matching_validation_ds.reset_format()
# Initialize the evaluator
dev_evaluator1 = BinaryClassificationEvaluator(
    sentences1=matching_validation_ds["sentence1"],
    sentences2=matching_validation_ds["sentence2"],
    labels=matching_validation_ds["label"],
    name="matching_eval",
)
dev_evaluator1(model)

train_loss = losses.ContrastiveLoss(model=model)

arg_batch_size = 256
arg_eval_steps = 1024
arg_saving_steps = 1024*2

wandb.init(project="Concept_Mapping", name=output_dir)  # Initialize here
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=arg_batch_size,
    per_device_eval_batch_size=arg_batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if your GPU can't handle FP16
    bf16=False,  # Set to True if your GPU supports BF16
    # batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=arg_eval_steps,
    save_strategy="no",
    save_steps=arg_saving_steps,
    save_total_limit=None,
    logging_steps=1024,
    # run_name=base_model,  # Used in W&B if `wandb` is installed
    report_to="wandb"
)

train_dataset = {
    'matching': iterable_matching,
    'offspring': iterable_offspring,
    'ancestor': iterable_ancestor
}
train_dataset = {k: train_dataset[k] for k in ds_names}

validation_dataset = {
    'matching': matching_validation_ds,
    'offspring': offspring_validation_ds,
    'ancestor': ancestor_validation_ds
}


print(train_dataset)
print(validation_dataset)


trainer = SentenceTransformerTrainer(
            model=model,
            train_dataset=None,
            eval_dataset=validation_dataset['matching'],
            loss=train_loss,
            args=args,
            evaluator=dev_evaluator1,
        )


## TODO: validate the sampling ratio gives the expected number of samples
max_saves = 4
block_size = arg_batch_size*arg_saving_steps*2
shuffle_buffer = arg_batch_size*arg_saving_steps*2
ratios = {'matching': 1, 'offspring': 0.5, 'ancestor': 0.5}
ratios = {k: ratios[k] for k in ds_names}
sampler = DictBatchSampler(train_dataset, batch_size=block_size, ratios = ratios, shuffle_buffer=shuffle_buffer)
ds_sizes = sampler.element_size()
iterations = sampler.iteration_size()

iterations_per_epoch = max(iterations.values())
epoch_num = 5
for i in range(epoch_num):
    if i!=0:
        trainer.args.warmup_ratio = 0.0
    j = 0
    for ds_train in sampler:
        j += 1
        print(f"Epoch {i+1}/{epoch_num}, Iteration {j}/{iterations_per_epoch}")
        trainer.train_dataset = ds_train
        trainer.train()
        auto_save_model(model, tokenizer, output_dir, max_saves=max_saves)

# auto_load_model(output_dir)
