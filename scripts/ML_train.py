from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from datasets import Dataset
dataset2 = Dataset.load_from_disk('data/train_dataset')

## trainint and testing split
dt = dataset2.train_test_split(test_size=0.001, seed=42)
train_dataset = dt['train']
test_dataset = dt['test']


model = SentenceTransformer('models/all-MiniLM-L6-v2')
train_loss = losses.ContrastiveLoss(model=model)


from sentence_transformers.training_args import SentenceTransformerTrainingArguments

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/fine-tuned",
    # Optional training parameters:
    num_train_epochs=10,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
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