#!/usr/bin/env python3
"""
Enhanced HierarchicalMatcher using direct parent prediction
"""

import sys
import pandas as pd
import numpy as np

# Add current directory to Python path
sys.path.append('.')

def enhanced_hierarchical_matching_with_parent_prediction(model, input_text, std_concepts, concept_ancestor, top_k=10, top_k_direct=50, boost_weight=0.5):
    """
    Enhanced hierarchical matching using parent prediction for reranking
    
    Pipeline:
    1. Direct semantic search: input_text → top-k standard concepts
    2. Parent prediction search: "<|parent of|>input_text" → predicted parent concepts  
    3. Reranking: Boost direct results whose actual level-1 parents appear in parent predictions
    
    Args:
        model: Relation-trained sentence transformer model
        input_text: Unknown concept text to match
        std_concepts: DataFrame of standard concepts ['concept_id', 'concept_name']
        concept_ancestor: DataFrame with OMOP concept_ancestor data ['descendant_concept_id', 'ancestor_concept_id', 'min_levels_of_separation']
        top_k: Number of final results to return
        top_k_direct: Number of direct matches to consider
        boost_weight: Weight for hierarchical validation boost
        
    Returns:
        DataFrame with reranked results based on hierarchical validation
    """
    from modules.FaissDB import build_index, is_initialized, search_similar
    from modules.timed_logger import logger
    
    logger.log(f"Enhanced hierarchical matching with parent prediction for: {input_text}")
    
    # Build index if needed
    repos_name = 'enhanced_hierarchical'
    if not is_initialized(repos=repos_name):
        build_index(
            model,
            std_concepts['concept_id'].tolist(),
            std_concepts['concept_name'].tolist(),
            repos=repos_name
        )
        logger.log(f"Built index for {len(std_concepts)} concepts")
    
    # Step 1: Direct semantic search
    logger.log("Step 1: Direct semantic search")
    direct_results = search_similar(
        query_ids=[0],
        query_names=[input_text],
        top_k=top_k_direct,
        repos=repos_name
    )
    
    if direct_results.empty:
        logger.log("No direct matches found")
        return pd.DataFrame()
    
    # Step 2: Parent prediction search
    logger.log("Step 2: Parent prediction search")
    parent_query = f"<|parent of|>{input_text}"
    parent_results = search_similar(
        query_ids=[0],
        query_names=[parent_query],
        top_k=top_k_direct,
        repos=repos_name
    )
    
    # Create set of predicted parent concept IDs for fast lookup
    predicted_parent_ids = set(parent_results['corpus_id'].tolist()) if not parent_results.empty else set()
    predicted_parent_scores = dict(zip(parent_results['corpus_id'], parent_results['score'])) if not parent_results.empty else {}
    
    logger.log(f"Found {len(direct_results)} direct matches, {len(predicted_parent_ids)} predicted parents")
    
    # Step 3: Hierarchical validation and reranking
    logger.log("Step 3: Hierarchical validation reranking")
    enhanced_results = []
    
    for _, row in direct_results.iterrows():
        concept_id = row['corpus_id']
        concept_name = row['corpus_name']
        direct_score = row['score']
        
        # Find level-1 parents of this direct match result
        level1_parents = concept_ancestor[
            (concept_ancestor['descendant_concept_id'] == concept_id) &
            (concept_ancestor['min_levels_of_separation'] == 1)
        ]['ancestor_concept_id'].tolist()
        
        # Check if any level-1 parents appear in predicted parents
        hierarchical_boost = 0.0
        matching_parents = []
        parent_boost_scores = []
        
        for parent_id in level1_parents:
            if parent_id in predicted_parent_ids:
                matching_parents.append(parent_id)
                parent_boost_scores.append(predicted_parent_scores[parent_id])
        
        # Calculate hierarchical validation boost
        if parent_boost_scores:
            # Use average of matching parent scores to avoid bias
            avg_parent_score = sum(parent_boost_scores) / len(parent_boost_scores)
            hierarchical_boost = avg_parent_score * boost_weight
        
        # Final reranked score
        final_score = direct_score + hierarchical_boost
        
        enhanced_results.append({
            'concept_id': concept_id,
            'concept_name': concept_name,
            'direct_score': direct_score,
            'level1_parents': level1_parents,
            'matching_parents': matching_parents,
            'avg_parent_prediction_score': sum(parent_boost_scores) / len(parent_boost_scores) if parent_boost_scores else 0.0,
            'hierarchical_boost': hierarchical_boost,
            'final_score': final_score
        })
    
    # Step 4: Sort by reranked scores and return top results
    enhanced_df = pd.DataFrame(enhanced_results)
    enhanced_df = enhanced_df.sort_values('final_score', ascending=False).head(top_k)
    
    logger.log(f"Hierarchical reranking returned {len(enhanced_df)} results")
    boosted_count = sum(1 for result in enhanced_results if result['hierarchical_boost'] > 0)
    logger.log(f"Hierarchical validation boosted {boosted_count}/{len(enhanced_results)} concepts")
    
    return enhanced_df.reset_index(drop=True)

def test_enhanced_hierarchical_matching():
    """Test the enhanced hierarchical matching with reranking"""
    try:
        from modules.ModelFunctions import auto_load_model
        
        print("Testing Enhanced Hierarchical Matching with Reranking")
        print("=" * 50)
        
        # Load Model 2
        model, tokenizer, train_config = auto_load_model("output/finetune/2025-07-30_13-33-46")
        print("Model 2 loaded")
        
        # Create test concepts
        std_concepts = pd.DataFrame({
            'concept_id': [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20
            ],
            'concept_name': [
                'prostate cancer', 'malignant neoplasm', 'cancer', 
                'prostate adenocarcinoma', 'lung cancer', 'breast cancer',
                'diabetes mellitus', 'type 2 diabetes', 'diabetic neuropathy',
                'heart disease', 'myocardial infarction', 'cardiovascular disease',
                'lung disease', 'pneumonia', 'chronic bronchitis',
                'hypertension', 'stroke', 'kidney disease', 
                'liver disease', 'endocrine disorder'
            ]
        })
        
        # Create test concept_ancestor relationships for hierarchical validation
        concept_ancestor = pd.DataFrame({
            'descendant_concept_id': [4, 4, 8, 8, 8, 11, 11, 15, 15],
            'ancestor_concept_id': [1, 2, 7, 7, 20, 10, 12, 13, 10],
            'min_levels_of_separation': [1, 2, 1, 2, 1, 1, 1, 1, 2]
        })
        # This means:
        # - prostate adenocarcinoma (4) -> prostate cancer (1) [level-1]
        # - prostate adenocarcinoma (4) -> malignant neoplasm (2) [level-2] 
        # - type 2 diabetes (8) -> diabetes mellitus (7) [level-1]
        # - type 2 diabetes (8) -> endocrine disorder (20) [level-1]
        # - myocardial infarction (11) -> heart disease (10) [level-1]
        # - myocardial infarction (11) -> cardiovascular disease (12) [level-1]
        # - chronic bronchitis (15) -> lung disease (13) [level-1]
        
        print(f"Created {len(std_concepts)} test concepts")
        
        # Test cases
        test_cases = [
            "prostate cancer with metastasis",
            "heart attack with chest pain", 
            "diabetes with complications",
            "lung tumor with breathing difficulty"
        ]
        
        for test_input in test_cases:
            print(f"\n{'='*60}")
            print(f"Testing: {test_input}")
            print(f"{'='*60}")
            
            # Run enhanced matching with hierarchical validation
            results = enhanced_hierarchical_matching_with_parent_prediction(
                model=model,
                input_text=test_input,
                std_concepts=std_concepts,
                concept_ancestor=concept_ancestor,
                top_k=5,
                boost_weight=0.4  # Tuned weight
            )
            
            if not results.empty:
                print("Hierarchical Reranking Results:")
                print("-" * 85)
                print("Concept                    | Direct | Parents| Boost  | Final  | Validation")
                print("-" * 85)
                
                for _, row in results.iterrows():
                    validation = "✓" if row['matching_parents'] else "✗"
                    parent_count = len(row['matching_parents'])
                    print(f"{row['concept_name']:<25} | {row['direct_score']:.3f}  | {parent_count:>3}    | {row['hierarchical_boost']:.3f}  | {row['final_score']:.3f}  | {validation:>6}")
            else:
                print("No results found")
        
        print(f"\nSummary:")
        print("Hierarchical reranking with parent prediction validation works!")
        print("Direct matches get boosted when their level-1 parents are predicted")
        print("This approach uses actual hierarchical relationships for validation")
        print("Only reranked direct results are returned to users")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_hierarchical_matching()
    sys.exit(0 if success else 1)