import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch
import pandas as pd
from datetime import datetime
import logging
import wandb
# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset
from modules.ML_data import get_matching, get_relation, get_matching_validation,get_relation_positive_validation, DictBatchSampler
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from modules.ML_train import get_base_model, auto_save_model, save_best_model, CustomEvaluator
from modules.TOKENS import TOKENS



## disable default huggingface logging
logging.getLogger("transformers").setLevel(logging.WARNING)  # Or logging.ERROR
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


## make sure cuda is available
torch.cuda.is_available()


base_model = 'all-MiniLM-L6-v2'
base_model_path = f'models/{base_model}'
# base_model = 'models/base_exclude_CIEL/checkpoint-65000'
output_dir = f"output/{base_model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

special_tokens = [TOKENS.child]
## what datasets to use?
ds_names = ['matching', 'ancestor']

## columns to use
cols = ['sentence1', 'sentence2', 'label']


n_neg_matching = 50
n_neg_relation = 50
n_fp_matching = 50
dt_seed = 42
data_folder = 'data/ML'
iterable_matching = get_matching(data_folder, n_neg=n_neg_matching, seed=dt_seed, n_fp=n_fp_matching)
iterable_offspring, iterable_ancestor = get_relation(data_folder, n_neg = n_neg_relation, seed=dt_seed)

reserved_concepts = pd.read_feather("data/ML/base_data/reserved_concepts.feather")
reserved_concepts.columns

nonstd_conditions = reserved_concepts[reserved_concepts['domain_id'] != 'Condition']


#################################
## For validation
#################################
conceptEX = pd.read_feather('data/omop_feather/conceptEX.feather')
conditions = conceptEX[conceptEX['domain_id'] == 'Condition']
std_conditions = conditions[conditions['standard_concept'] == 'S']
nonstd_conditions = conditions[conditions['standard_concept'] != 'S']

valid_size = 1000
matching_validation_pd = nonstd_conditions.sample(valid_size, random_state=dt_seed)


database = std_conditions[['concept_id', 'concept_name']]
query = nonstd_conditions[['concept_id', 'concept_name', 'std_concept_id']][nonstd_conditions['vocabulary_id'] == 'CIEL']



evaluator = CustomEvaluator(reference=database, query=query, n_results=100)



model, tokenizer = get_base_model(base_model_path, special_tokens)

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
best_eval_loss = float('inf')

j = 0
for ds_train in sampler:
    epoch_i = j // iterations_per_epoch
    if epoch_i!=0:
        trainer.args.warmup_ratio = 0.0
    if epoch_i >= epoch_num:
        break
    ## How many iterations per evaluation?
    evaluator.build_reference(model=model)
    
    print(f"Epoch {epoch_i+1}/{epoch_num}, Iteration {j}/{iterations_per_epoch}")
    trainer.train_dataset = ds_train
    
    
    trainer.train()
    trainer.state
    auto_save_model(model, tokenizer, output_dir, max_saves=max_saves)
    eval_results = trainer.evaluate()
    if eval_results['eval_loss'] < best_eval_loss:
        best_eval_loss = eval_results['eval_loss']
        save_best_model(model, tokenizer, output_dir)
    j += 1
        
        

# auto_load_model(output_dir)
