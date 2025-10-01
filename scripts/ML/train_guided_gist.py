import os
import argparse
from datetime import datetime

import pandas as pd
from datasets import Dataset

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from modules.metrics import evaluate_embedding_similarity_with_mrr

from modules.ModelFunctions import get_ST_model, auto_load_model
from modules.timed_logger import logger


def build_positive_pairs(matching_base_path: str, relation_base_path: str, use_relation: bool, max_samples: int = None) -> Dataset:
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

    # Limit matching pairs if specified
    if max_samples is not None:
        match_pairs = match_pairs.sample(n=min(max_samples, len(match_pairs)), random_state=42).reset_index(drop=True)
        logger.log(f"Limited matching pairs to {len(match_pairs)} samples")

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

            # Limit relation pairs if specified
            if max_samples is not None:
                rel_samples = min(max_samples // 2, len(rel_pairs))  # Split between matching and relation
                rel_pairs = rel_pairs.sample(n=rel_samples, random_state=42).reset_index(drop=True)
                logger.log(f"Limited relation pairs to {len(rel_pairs)} samples")

            all_pairs = pd.concat([all_pairs, rel_pairs], ignore_index=True).drop_duplicates()

    return Dataset.from_pandas(all_pairs.reset_index(drop=True), preserve_index=False)


def main():
    parser = argparse.ArgumentParser(description='Guided in-batch training with GISTEmbedLoss / CachedGISTEmbedLoss')
    parser.add_argument('--guide-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='HF model name or local path for guide')
    parser.add_argument('--base-checkpoint', type=str, default='none', help='Path to load checkpoint (default: none - use sentence-transformers/all-MiniLM-L6-v2)')
    parser.add_argument('--use-cached', action='store_true', help='Use CachedGISTEmbedLoss for large effective batches')
    parser.add_argument('--temperature', type=float, default=0.01, help='GIST temperature')
    parser.add_argument('--mini-batch-size', type=int, default=64, help='Mini-batch size inside CachedGIST (ignored if not cached)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--per-device-train-batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no-relation', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--max-training-samples', type=int, default=200, help='Limit total training samples (default: 200)')
    args = parser.parse_args()

    logger.reset_timer()
    logger.log('Loading models for GIST training')

    # Load train model: checkpoint if specified, else use sentence-transformers/all-MiniLM-L6-v2
    if args.base_checkpoint != 'none' and os.path.exists(args.base_checkpoint):
        model, tokenizer, _ = auto_load_model(args.base_checkpoint)
    else:
        logger.log('Loading base model: sentence-transformers/all-MiniLM-L6-v2 with special tokens')
        model, tokenizer = get_ST_model('sentence-transformers/all-MiniLM-L6-v2')

    # Load guide model (kept frozen by the loss)
    guide = SentenceTransformer(args.guide_model)

    # Build positives-only dataset
    matching_base_path = 'data/matching'
    relation_base_path = 'data/relation'
    use_relation = not args.no_relation
    train_dataset = build_positive_pairs(matching_base_path, relation_base_path, use_relation, args.max_training_samples)

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
        eval_strategy='no',  # Changed from evaluation_strategy
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

    # Evaluate MRR on OMOP CIEL validation and other available sets
    matching_base_path = 'data/matching'
    relation_base_path = 'data/relation'

    metrics_out = {}

    # OMOP CIEL validation (primary)
    try:
        from modules.ChromaVecDB import ChromaVecDB

        logger.log('Loading OMOP CIEL validation data')
        conceptEX = pd.read_feather('data/omop_feather/conceptEX.feather')
        conditions = conceptEX[conceptEX['domain_id'] == 'Condition']
        std_conditions = conditions[conditions['standard_concept'] == 'S']
        nonstd_conditions = conditions[conditions['standard_concept'] != 'S']

        # Database: standard conditions
        database = std_conditions[['concept_id', 'concept_name']]

        # Query: CIEL concepts
        query_df = nonstd_conditions[
            ['concept_id', 'concept_name', 'std_concept_id']
        ][nonstd_conditions['vocabulary_id'] == 'CIEL']

        # Create vector database and query
        db = ChromaVecDB(model=model, name='ciel_eval', path=None)
        db.empty_collection()
        db.store_concepts(database, batch_size=5461)

        max_k = 50
        res = db.query(query_df[['concept_name']], n_results=max_k)

        y_true = query_df['std_concept_id'].str[0].astype(int).tolist()
        pred_lists = res['ids']

        # Calculate comprehensive metrics
        k_list = [1, 10, 50]
        ciel_metrics = {}

        for k in k_list:
            hits = [1 if (y_true[i] in pred_lists[i][:k]) else 0 for i in range(len(y_true))]
            acc_k = sum(hits) / len(hits)
            ciel_metrics[f'Accuracy@{k}'] = acc_k

        # Calculate MRR
        reciprocal_ranks = []
        for true_id, preds in zip(y_true, pred_lists):
            try:
                rank = preds.index(true_id) + 1
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        ciel_metrics['MRR'] = mrr

        metrics_out['omop_ciel'] = ciel_metrics
        print('OMOP CIEL validation MRR:', mrr)
        print('OMOP CIEL validation Accuracy@1:', ciel_metrics['Accuracy@1'])

    except Exception as e:
        print(f'Warning: Could not load OMOP CIEL validation: {e}')

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

    # Matching valid (fallback)
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
