import os
import argparse
from datetime import datetime

import pandas as pd
from datasets import Dataset

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from modules.metrics import evaluate_embedding_similarity_with_mrr

from modules.ModelFunctions import get_ST_model, auto_load_model
from modules.timed_logger import logger


def build_positive_pairs(matching_base_path: str, relation_base_path: str, use_relation: bool) -> Dataset:
    # Load matching tables
    name_table = pd.read_feather(os.path.join(matching_base_path, 'condition_matching_name_table_train.feather'))
    name_bridge = pd.read_feather(os.path.join(matching_base_path, 'condition_matching_name_bridge_train.feather'))

    # Target concepts: prefer std_condition_concept, fallback to target_concepts
    target_concepts_path = os.path.join(matching_base_path, 'std_condition_concept.feather')
    if not os.path.exists(target_concepts_path):
        alt_path = os.path.join(matching_base_path, 'target_concepts.feather')
        target_concepts_path = alt_path if os.path.exists(alt_path) else target_concepts_path
    target_concepts = pd.read_feather(target_concepts_path)

    match_pairs = name_bridge.merge(
        name_table[['name_id', 'name']].rename(columns={'name': 'anchor'}),
        on='name_id'
    ).merge(
        target_concepts[['concept_id', 'concept_name']].rename(columns={'concept_name': 'positive'}),
        on='concept_id'
    )[['anchor', 'positive']].drop_duplicates()

    all_pairs = match_pairs

    # Optionally add relation positives
    if use_relation:
        rel_table_path = os.path.join(relation_base_path, 'name_table_relation.feather')
        rel_bridge_path = os.path.join(relation_base_path, 'name_bridge_relation.feather')
        if os.path.exists(rel_table_path) and os.path.exists(rel_bridge_path):
            name_table_rel = pd.read_feather(rel_table_path)
            name_bridge_rel = pd.read_feather(rel_bridge_path)
            rel_pairs = name_bridge_rel.merge(
                name_table_rel[['name_id', 'name']].rename(columns={'name': 'anchor'}),
                on='name_id'
            ).merge(
                target_concepts[['concept_id', 'concept_name']].rename(columns={'concept_name': 'positive'}),
                on='concept_id'
            )[['anchor', 'positive']].drop_duplicates()
            all_pairs = pd.concat([all_pairs, rel_pairs], ignore_index=True).drop_duplicates()

    return Dataset.from_pandas(all_pairs.reset_index(drop=True), preserve_index=False)


def main():
    parser = argparse.ArgumentParser(description='Guided in-batch training with GISTEmbedLoss / CachedGISTEmbedLoss')
    parser.add_argument('--guide-model', type=str, default='all-MiniLM-L6-v2', help='HF model name or local path for guide')
    parser.add_argument('--base-checkpoint', type=str, default='output/finetune/2025-07-28_23-20-24', help='Path to load latest fine-tuned checkpoint; falls back to ClinicalBERT ST')
    parser.add_argument('--use-cached', action='store_true', help='Use CachedGISTEmbedLoss for large effective batches')
    parser.add_argument('--temperature', type=float, default=0.01, help='GIST temperature')
    parser.add_argument('--mini-batch-size', type=int, default=64, help='Mini-batch size inside CachedGIST (ignored if not cached)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--per-device-train-batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no-relation', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    logger.reset_timer()
    logger.log('Loading models for GIST training')

    # Load train model: latest checkpoint if available, else base ST
    model, tokenizer, _ = auto_load_model(args.base_checkpoint) if os.path.exists(args.base_checkpoint) else (None, None, None)
    if model is None:
        model, tokenizer = get_ST_model('ClinicalBERT')

    # Load guide model (kept frozen by the loss)
    guide = SentenceTransformer(args.guide_model)

    # Build positives-only dataset
    matching_base_path = 'data/matching'
    relation_base_path = 'data/relation'
    use_relation = not args.no_relation
    train_dataset = build_positive_pairs(matching_base_path, relation_base_path, use_relation)

    # Loss
    if args.use_cached:
        loss = losses.CachedGISTEmbedLoss(
            model=model,
            guide=guide,
            temperature=args.temperature,
            mini_batch_size=args.mini_batch_size,
            show_progress_bar=True,
        )
    else:
        loss = losses.GISTEmbedLoss(
            model=model,
            guide=guide,
            temperature=args.temperature,
        )

    # Output dir
    output_dir = args.output_dir or f"output/guided_gist/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        logging_steps=200,
        save_steps=1000,
        save_total_limit=2,
        report_to=[],  # disable HF Hub/W&B by default
        evaluation_strategy='no',
        do_train=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    trainer.train()

    # Save final model
    model.save(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate MRR on available validation sets
    matching_base_path = 'data/matching'
    relation_base_path = 'data/relation'

    def _normalize_eval_df(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # Rename to expected columns if needed
        if 'corpus_name' not in d.columns and 'sentence1' in d.columns:
            d = d.rename(columns={'sentence1': 'corpus_name'})
        if 'query_name' not in d.columns and 'sentence2' in d.columns:
            d = d.rename(columns={'sentence2': 'query_name'})
        # Ensure IDs for grouping
        if 'query_id' not in d.columns:
            d['query_id'] = pd.factorize(d['query_name'])[0]
        if 'corpus_id' not in d.columns:
            d['corpus_id'] = pd.factorize(d['corpus_name'])[0]
        return d[['corpus_name','query_name','corpus_id','query_id','label']]

    metrics_out = {}
    # Matching valid
    valid_path = os.path.join(matching_base_path, 'condition_matching_valid.feather')
    if os.path.exists(valid_path):
        df_valid = pd.read_feather(valid_path)
        df_valid = _normalize_eval_df(df_valid)
        m_valid = evaluate_embedding_similarity_with_mrr(model, df_valid)
        metrics_out['eval'] = m_valid
        print('Matching valid MRR:', m_valid.get('reciprocal_rank'))

    # Matching train subset
    train_subset_path = os.path.join(matching_base_path, 'condition_matching_train_subset.feather')
    if os.path.exists(train_subset_path):
        df_train_sub = pd.read_feather(train_subset_path)
        df_train_sub = _normalize_eval_df(df_train_sub)
        m_train = evaluate_embedding_similarity_with_mrr(model, df_train_sub)
        metrics_out['train_matching'] = m_train
        print('Matching train subset MRR:', m_train.get('reciprocal_rank'))

    # Relation train subset (optional)
    rel_subset_path = os.path.join(relation_base_path, 'condition_relation_train_subset.feather')
    if os.path.exists(rel_subset_path):
        df_rel_sub = pd.read_feather(rel_subset_path)
        df_rel_sub = _normalize_eval_df(df_rel_sub)
        m_rel = evaluate_embedding_similarity_with_mrr(model, df_rel_sub)
        metrics_out['train_relation'] = m_rel
        print('Relation train subset MRR:', m_rel.get('reciprocal_rank'))

    # Persist metrics
    try:
        import json
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_out, f, indent=2)
    except Exception as e:
        print('Warning: failed to save metrics.json:', e)

    logger.done()


if __name__ == '__main__':
    main()
