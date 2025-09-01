#!/usr/bin/env python3
"""
Improved Enhanced Algorithm with fixes for the identified issues
"""

import sys
import pandas as pd
import numpy as np
sys.path.append('.')

from typing import List, Tuple

def get_improved_enhanced_model_top_k_predictions(baseline_model, relation_model, query: str, concept_corpus: pd.DataFrame, 
                                                 concept_ancestor: pd.DataFrame, top_k: int = 50) -> List[Tuple[int, str, float]]:
    """
    Improved enhanced algorithm with fixes for:
    1. Too high thresholds
    2. Self-match dominance
    3. Better adaptive weighting
    4. Focus on meaningful hierarchical signals
    """
    
    # 1. Baseline matching
    baseline_embeddings = baseline_model.encode([query], normalize_embeddings=True)
    corpus_embeddings_baseline = baseline_model.encode(concept_corpus['concept_name'].tolist(), normalize_embeddings=True)
    baseline_similarities = np.dot(baseline_embeddings, corpus_embeddings_baseline.T).flatten()
    
    # 2. Remove perfect self-matches to allow other concepts to compete
    perfect_match_indices = np.where(baseline_similarities >= 0.999)[0]
    query_concept_ids = set()
    for idx in perfect_match_indices:
        query_concept_ids.add(concept_corpus.iloc[idx]['concept_id'])
    
    print(f"🔹 Found {len(perfect_match_indices)} perfect self-matches, will apply special handling")
    
    # 3. Parent prediction matching  
    parent_query = f"<|parent of|>{query}"
    parent_query_emb = relation_model.encode([parent_query], normalize_embeddings=True)
    
    # 4. Build parent lookup for efficient hierarchical validation
    unique_corpus_ids = set(concept_corpus['concept_id'].unique())
    relevant_relationships = concept_ancestor[
        (concept_ancestor['min_levels_of_separation'] == 1) &
        (concept_ancestor['descendant_concept_id'].isin(unique_corpus_ids))
    ][['descendant_concept_id', 'ancestor_concept_id']]
    
    parent_lookup = {}
    for _, row in relevant_relationships.iterrows():
        descendant = row['descendant_concept_id']
        ancestor = row['ancestor_concept_id']
        if descendant not in parent_lookup:
            parent_lookup[descendant] = []
        parent_lookup[descendant].append(ancestor)
    
    # Create concept_id to concept_name mapping
    concept_id_to_name = dict(zip(concept_corpus['concept_id'], concept_corpus['concept_name']))
    
    # Pre-compute parent embeddings
    all_parent_ids = set()
    for descendant_id in unique_corpus_ids:
        parents = parent_lookup.get(descendant_id, [])
        all_parent_ids.update(parents)
    
    parent_id_list = []
    parent_name_list = []
    for parent_id in all_parent_ids:
        if parent_id in concept_id_to_name:
            parent_id_list.append(parent_id)
            parent_name_list.append(concept_id_to_name[parent_id])
    
    if parent_name_list:
        parent_embeddings_batch = relation_model.encode(parent_name_list, normalize_embeddings=True)
        parent_id_to_embedding = dict(zip(parent_id_list, parent_embeddings_batch))
    else:
        parent_id_to_embedding = {}
    
    # 5. Apply improved hierarchical enhancement
    enhanced_similarities = baseline_similarities.copy()
    
    # IMPROVED PARAMETERS - Much more permissive thresholds
    boost_weight_high = 0.3       # Increased from 0.2
    boost_weight_medium = 0.15    # Increased from 0.1
    boost_weight_low = 0.08       # New tier for weak signals
    penalty_factor = 0.92         # Stronger penalty
    
    # LOWERED THRESHOLDS - More realistic for semantic similarity
    high_threshold = 0.5          # Lowered from 0.8
    medium_threshold = 0.35       # Lowered from 0.6  
    low_threshold = 0.2           # New low tier
    negative_threshold = 0.15     # Lowered from 0.3
    
    boost_count_high = 0
    boost_count_medium = 0
    boost_count_low = 0
    penalty_count = 0
    self_match_boosts = 0
    
    print(f"🔹 IMPROVED thresholds: High≥{high_threshold}, Med≥{medium_threshold}, Low≥{low_threshold}, Neg<{negative_threshold}")
    
    for i, row in concept_corpus.iterrows():
        corpus_id = row['concept_id']
        baseline_score = baseline_similarities[i]
        
        # Find level-1 parents
        level1_parents = parent_lookup.get(corpus_id, [])
        
        if level1_parents:
            # Get pre-computed parent embeddings
            parent_embeddings = []
            for parent_id in level1_parents:
                if parent_id in parent_id_to_embedding:
                    parent_embeddings.append(parent_id_to_embedding[parent_id])
            
            if parent_embeddings:
                # Compute hierarchical validation strength
                max_parent_match_similarity = 0.0
                
                for parent_emb in parent_embeddings:
                    parent_emb_single = parent_emb.reshape(1, -1)
                    similarity = np.dot(parent_query_emb, parent_emb_single.T).flatten()[0]
                    max_parent_match_similarity = max(max_parent_match_similarity, similarity)
                
                # IMPROVED ADAPTIVE WEIGHTING
                # More aggressive boosts for lower baseline scores
                if baseline_score < 0.3:  # Very low confidence
                    adaptive_boost_high = boost_weight_high * 2.0      # Double boost
                    adaptive_boost_medium = boost_weight_medium * 1.8
                    adaptive_boost_low = boost_weight_low * 1.6
                elif baseline_score < 0.6:  # Medium confidence
                    adaptive_boost_high = boost_weight_high * 1.5      # 1.5x boost
                    adaptive_boost_medium = boost_weight_medium * 1.3
                    adaptive_boost_low = boost_weight_low * 1.2
                elif baseline_score < 0.9:  # High confidence
                    adaptive_boost_high = boost_weight_high           # Normal boost
                    adaptive_boost_medium = boost_weight_medium
                    adaptive_boost_low = boost_weight_low
                else:  # Very high confidence (near perfect matches)
                    # Special handling for self-matches - smaller but meaningful boosts
                    adaptive_boost_high = boost_weight_high * 0.5
                    adaptive_boost_medium = boost_weight_medium * 0.5
                    adaptive_boost_low = boost_weight_low * 0.5
                
                # Apply IMPROVED multi-tier enhancement logic
                original_score = enhanced_similarities[i]
                
                if max_parent_match_similarity >= high_threshold:
                    # Strong hierarchical validation
                    hierarchical_boost = max_parent_match_similarity * adaptive_boost_high
                    enhanced_similarities[i] += hierarchical_boost
                    boost_count_high += 1
                    
                    if baseline_score >= 0.999:  # Track self-match boosts
                        self_match_boosts += 1
                        
                elif max_parent_match_similarity >= medium_threshold:
                    # Medium hierarchical validation  
                    hierarchical_boost = max_parent_match_similarity * adaptive_boost_medium
                    enhanced_similarities[i] += hierarchical_boost
                    boost_count_medium += 1
                    
                elif max_parent_match_similarity >= low_threshold:
                    # NEW: Weak but positive hierarchical signal
                    hierarchical_boost = max_parent_match_similarity * adaptive_boost_low
                    enhanced_similarities[i] += hierarchical_boost
                    boost_count_low += 1
                    
                elif max_parent_match_similarity < negative_threshold:
                    # Negative validation → penalty
                    enhanced_similarities[i] *= penalty_factor
                    penalty_count += 1
                
                # NO CHANGE for middle range (0.15-0.2) - neutral zone
    
    print(f"🔹 IMPROVED Enhancement statistics:")
    print(f"   High boosts (≥{high_threshold}): {boost_count_high}")
    print(f"   Medium boosts (≥{medium_threshold}): {boost_count_medium}")  
    print(f"   Low boosts (≥{low_threshold}): {boost_count_low}")
    print(f"   Penalties (<{negative_threshold}): {penalty_count}")
    print(f"   Self-match boosts: {self_match_boosts}")
    print(f"   Total changes: {boost_count_high + boost_count_medium + boost_count_low + penalty_count}")
    
    # 6. Enhanced ranking with smart self-match handling
    top_k_indices = np.argsort(enhanced_similarities)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_k_indices):
        concept_id = concept_corpus.iloc[idx]['concept_id']
        concept_name = concept_corpus.iloc[idx]['concept_name']
        score = enhanced_similarities[idx]
        results.append((concept_id, concept_name, score))
    
    return results

def test_improved_algorithm():
    """
    Test the improved algorithm on a few queries
    """
    
    from modules.ModelFunctions import auto_load_model
    from fair_model_comparison import get_model_top_k_predictions
    
    print("🚀 Testing Improved Enhanced Algorithm")
    print("=" * 60)
    
    # Load data and models
    queries_data = pd.read_feather('data/matching/condition_matching_test_pos_dedup.feather')
    test_queries = queries_data.head(3)
    
    model1, _, _ = auto_load_model("output/matching_model")
    model2, _, _ = auto_load_model("output/finetune/2025-07-30_13-33-46")
    
    concepts1 = queries_data[['concept_id1', 'sentence1']].rename(columns={'concept_id1': 'concept_id', 'sentence1': 'concept_name'})
    concepts2 = queries_data[['concept_id2', 'sentence2']].rename(columns={'concept_id2': 'concept_id', 'sentence2': 'concept_name'})
    concept_corpus = pd.concat([concepts1, concepts2]).drop_duplicates('concept_id').reset_index(drop=True)
    
    concept_ancestor = pd.read_feather('data/omop_feather/concept_ancestor.feather')
    concept_ancestor_level1 = concept_ancestor[concept_ancestor['min_levels_of_separation'] == 1]
    
    # Test each query
    for idx, query_row in test_queries.iterrows():
        query = query_row['sentence1']
        true_concept_id = query_row['concept_id2']
        
        print(f"\n🔍 TESTING QUERY: '{query}'")
        print(f"   True target: '{query_row['sentence2']}' (ID: {true_concept_id})")
        print("-" * 60)
        
        # Get baseline vs improved enhanced predictions
        baseline_preds = get_model_top_k_predictions(model1, query, concept_corpus, top_k=10)
        improved_preds = get_improved_enhanced_model_top_k_predictions(
            model1, model2, query, concept_corpus, concept_ancestor_level1, top_k=10
        )
        
        # Find true target ranks
        baseline_rank = None
        improved_rank = None
        
        for rank, (pred_id, _, _) in enumerate(baseline_preds):
            if pred_id == true_concept_id:
                baseline_rank = rank + 1
                break
        
        for rank, (pred_id, _, _) in enumerate(improved_preds):
            if pred_id == true_concept_id:
                improved_rank = rank + 1
                break
        
        print(f"📊 TRUE TARGET RANKING:")
        print(f"   Baseline rank: {baseline_rank if baseline_rank else 'Not in top-10'}")
        print(f"   Improved rank: {improved_rank if improved_rank else 'Not in top-10'}")
        
        if baseline_rank and improved_rank:
            improvement = baseline_rank - improved_rank
            if improvement > 0:
                print(f"   ✅ IMPROVEMENT: +{improvement} positions!")
            elif improvement < 0:
                print(f"   ❌ DEGRADATION: {improvement} positions")
            else:
                print(f"   ➡️  NO CHANGE")
        
        print(f"\n📈 Top-5 Comparison:")
        print(f"{'Rank':<4} {'Baseline':<15} {'Improved':<15} {'Concept':<45}")
        print("-" * 85)
        
        for rank in range(min(5, len(baseline_preds), len(improved_preds))):
            b_id, b_name, b_score = baseline_preds[rank]
            i_id, i_name, i_score = improved_preds[rank]
            
            marker = "🎯" if b_id == true_concept_id else "  "
            print(f"{marker}{rank+1:<3} {b_score:<15.4f} {i_score:<15.4f} {b_name[:40]:<45}")

if __name__ == "__main__":
    test_improved_algorithm()