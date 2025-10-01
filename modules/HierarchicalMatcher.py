"""
Hierarchical Concept Matching

improve concept matching accuracy by using hierarchical relationship information from OMOP concept_ancestor data.

The core idea is to enhance pure semantic similarity with hierarchical knowledge:
- Use concept_ancestor relationships to identify parent/child connections
- Combine semantic + hierarchical scores for better matching accuracy

"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from modules.FaissDB import search_similar
from modules.timed_logger import logger


class HierarchicalMatcher:
    """
    Enhanced concept matcher that combines semantic similarity with hierarchical relationships
    """
    
    def __init__(self, concept_ancestor: pd.DataFrame, concept_table: pd.DataFrame):
        """
        Initialize the hierarchical matcher
        
        Args:
            concept_ancestor: DataFrame with columns ['ancestor_concept_id', 'descendant_concept_id', 'min_levels_of_separation', 'max_levels_of_separation']
            concept_table: DataFrame with columns ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id']
        """
        self.concept_ancestor = concept_ancestor
        self.concept_table = concept_table
        
        # Create lookup dictionaries for fast access
        self._build_hierarchy_maps()
        self._build_concept_lookup()
        
    def _build_hierarchy_maps(self):
        """Build fast lookup maps for hierarchical relationships"""
        logger.log(f"Processing {len(self.concept_ancestor)} ancestor relationships")
        
        # Build parent-to-children mapping using groupby 
        parent_groups = self.concept_ancestor.groupby('ancestor_concept_id')['descendant_concept_id'].apply(list)
        self.parent_to_children = parent_groups.to_dict()
        
        # Build child-to-parents mapping using groupby
        child_groups = self.concept_ancestor.groupby('descendant_concept_id')['ancestor_concept_id'].apply(list)
        self.child_to_parents = child_groups.to_dict()
        
        # Build distance mapping using vectorized operations
        self.hierarchy_distances = dict(zip(
            zip(self.concept_ancestor['ancestor_concept_id'], self.concept_ancestor['descendant_concept_id']),
            self.concept_ancestor['min_levels_of_separation']
        ))
        
        logger.log(f"Built {len(self.parent_to_children)} parent->children mappings")
        logger.log(f"Built {len(self.child_to_parents)} child->parent mappings") 
        logger.log(f"Built {len(self.hierarchy_distances)} distance mappings")
            
    def _build_concept_lookup(self):
        """Build fast concept lookup by ID"""
        self.concept_lookup = self.concept_table.set_index('concept_id').to_dict('index')
        
    def get_hierarchical_distance(self, concept_id1: int, concept_id2: int) -> Optional[float]:
        """
        Calculate hierarchical distance between two concepts
        
        Args:
            concept_id1: First concept ID
            concept_id2: Second concept ID
            
        Returns:
            Hierarchical distance (lower = closer relationship), None if no relationship
        """
        # Check direct parent-child relationship
        if (concept_id1, concept_id2) in self.hierarchy_distances:
            return self.hierarchy_distances[(concept_id1, concept_id2)]
        if (concept_id2, concept_id1) in self.hierarchy_distances:
            return self.hierarchy_distances[(concept_id2, concept_id1)]
            
        # Check if concepts are siblings (share common parent)
        parents1 = set(self.child_to_parents.get(concept_id1, []))
        parents2 = set(self.child_to_parents.get(concept_id2, []))
        common_parents = parents1.intersection(parents2)
        
        if common_parents:
            # Sibling relationship - return average distance through common parent
            distances = []
            for parent in common_parents:
                dist1 = self.hierarchy_distances.get((parent, concept_id1), float('inf'))
                dist2 = self.hierarchy_distances.get((parent, concept_id2), float('inf'))
                if dist1 != float('inf') and dist2 != float('inf'):
                    distances.append(dist1 + dist2)
            if distances:
                return min(distances)
        
        return None
    
    def expand_search_candidates(self, concept_id: int, max_hierarchy_levels: int = 2) -> List[int]:
        """
        Expand search space to include hierarchically related concepts
        
        Args:
            concept_id: Base concept ID to expand from
            max_hierarchy_levels: Maximum levels of hierarchy to include
            
        Returns:
            List of concept IDs including original + hierarchically related concepts
        """
        candidates = {concept_id}  
        
        candidates.update(self.child_to_parents.get(concept_id, []))
        candidates.update(self.parent_to_children.get(concept_id, []))
        
        if max_hierarchy_levels > 1:
            current_level = {concept_id}
            for level in range(2, max_hierarchy_levels + 1):
                next_level = set()
                for current_concept in current_level:
                    # Add parents and children of current level concepts
                    next_level.update(self.child_to_parents.get(current_concept, []))
                    next_level.update(self.parent_to_children.get(current_concept, []))
                
                # Remove already found candidates to avoid duplicates
                next_level -= candidates
                candidates.update(next_level)
                current_level = next_level
                
                # Stop if no new concepts found
                if not next_level:
                    break
                    
        return list(candidates)
    
    def calculate_combined_score(self, 
                               semantic_score: float, 
                               query_concept_id: int, 
                               candidate_concept_id: int,
                               semantic_weight: float = 0.7,
                               hierarchy_weight: float = 0.3,
                               domain_bonus: float = 0.1) -> float:
        """
        Calculate combined semantic + hierarchical score
        
        Args:
            semantic_score: Original embedding similarity score
            query_concept_id: ID of query concept
            candidate_concept_id: ID of candidate match concept  
            semantic_weight: Weight for semantic similarity (default 0.7)
            hierarchy_weight: Weight for hierarchical similarity (default 0.3)
            domain_bonus: Bonus for same-domain concepts (default 0.1)
            
        Returns:
            Combined score (higher = better match)
        """
        # Start with weighted semantic score
        combined_score = semantic_score * semantic_weight
        
        # Add hierarchical component
        hierarchical_distance = self.get_hierarchical_distance(query_concept_id, candidate_concept_id)
        
        if hierarchical_distance is not None:
            # Convert distance to similarity (closer = higher score)
            decay_factor = 2.0
            # use expenential decay for hierarchical similarity
            hierarchical_similarity = np.exp(-hierarchical_distance / decay_factor)
            combined_score += hierarchical_similarity * hierarchy_weight
            
            # Special bonus for direct parent-child relationships (distance = 1)
            if hierarchical_distance == 1:
                combined_score += 0.1
        
        # Domain bonus if concepts are in same domain
        if query_concept_id in self.concept_lookup and candidate_concept_id in self.concept_lookup:
            query_domain = self.concept_lookup[query_concept_id].get('domain_id', '')
            candidate_domain = self.concept_lookup[candidate_concept_id].get('domain_id', '')
            if query_domain == candidate_domain and query_domain != '':
                combined_score += domain_bonus
                
        return combined_score
    
    def preprocess_concepts(self):
        """简化且有效的concept预处理 - 允许重叠"""
        
        # Target: 所有concepts都可以作为匹配目标
        target_concepts = set(self.concept_table['concept_id'])
        
        # Boost: 所有作为ancestor的concepts都可以提供boost
        boost_concepts = set(self.concept_ancestor['ancestor_concept_id'])
        
        # 统计信息
        overlap_concepts = target_concepts & boost_concepts
        
        logger.log(f"Target concepts: {len(target_concepts)}")
        logger.log(f"Boost concepts: {len(boost_concepts)}")
        logger.log(f"Overlap concepts: {len(overlap_concepts)} (this is normal and expected)")
        
        return target_concepts, boost_concepts
    
    def find_level1_parents(self, concept_id: int) -> List[int]:
        """找到concept的直接父概念（level 1 parents only）"""
        level1_parents = self.concept_ancestor[
            (self.concept_ancestor['descendant_concept_id'] == concept_id) &
            (self.concept_ancestor['min_levels_of_separation'] == 1)
        ]['ancestor_concept_id'].tolist()
        return level1_parents
    
    def get_hierarchy_distance(self, ancestor_id: int, descendant_id: int) -> int:
        """Get hierarchy distance between ancestor and descendant"""
        relation = self.concept_ancestor[
            (self.concept_ancestor['ancestor_concept_id'] == ancestor_id) & 
            (self.concept_ancestor['descendant_concept_id'] == descendant_id)
        ]
        if not relation.empty:
            return relation['min_levels_of_separation'].iloc[0]
        return float('inf')
    
    def get_concept_name(self, concept_id: int) -> str:
        """获取concept名称"""
        concept_row = self.concept_table[self.concept_table['concept_id'] == concept_id]
        return concept_row['concept_name'].iloc[0] if not concept_row.empty else "Unknown"
    
    def hierarchical_matching_level1_parents(self,
                                           model,
                                           input_text: str,
                                           std_concepts: pd.DataFrame,
                                           top_k: int = 10,
                                           top_k_direct: int = 50,
                                           boost_weight: float = 0.5,
                                           repos: str = 'hierarchical_matching') -> pd.DataFrame:
        """
        Level 1 Parent Hierarchical Matching Algorithm
        
        Pipeline:
        1. Unknown concept -> top X direct matching results
        2. For each direct match, find its level 1 parents from concept_ancestor  
        3. Calculate similarity between input and these level 1 parents
        4. Average parent scores to prevent bias from concepts with many parents
        5. Add averaged parent score back to direct matching score
        
        Args:
            model: Trained sentence transformer model
            input_text: Unknown concept text to match
            std_concepts: DataFrame of standard concepts ['concept_id', 'concept_name']
            top_k: Number of final results to return
            top_k_direct: Number of direct matches to consider
            boost_weight: Weight for parent concept contribution (default: 0.5)
            repos: Repository name for FAISS index
            
        Returns:
            DataFrame with columns: ['concept_id', 'concept_name', 'direct_score', 'parent_boost', 'parent_count', 'final_score']
        """
        logger.log(f"Starting level 1 parent hierarchical matching for: {input_text}")
        
        # Step 1: Get direct semantic matches
        logger.log("Step 1: Direct semantic matching")
        direct_results = search_similar(
            query_ids=[0],  # Dummy ID for text search
            query_names=[input_text],
            top_k=top_k_direct,
            repos=repos
        )
        
        if direct_results.empty:
            logger.log("No direct semantic matches found")
            return pd.DataFrame()
        
        logger.log(f"Found {len(direct_results)} direct matches")
        
        # Step 2: For each direct match, find level 1 parents and calculate parent similarities
        logger.log("Step 2: Calculating parent similarities")
        enhanced_results = []
        
        for _, row in direct_results.iterrows():
            concept_id = row['corpus_id']
            direct_score = row['score']
            concept_name = row['corpus_name']
            
            # Find level 1 parents for this concept
            level1_parents = self.find_level1_parents(concept_id)
            
            # Calculate similarity between input and each parent
            parent_scores = []
            if level1_parents:
                # Get parent names
                parent_names = []
                valid_parent_ids = []
                
                for parent_id in level1_parents:
                    parent_name = self.get_concept_name(parent_id)
                    if parent_name != "Unknown":
                        parent_names.append(parent_name)
                        valid_parent_ids.append(parent_id)
                
                # Calculate similarities with parents
                if parent_names:
                    parent_embeddings = model.encode(parent_names, normalize_embeddings=True)
                    input_embedding = model.encode([input_text], normalize_embeddings=True)
                    
                    # Calculate cosine similarities
                    import numpy as np
                    similarities = np.dot(input_embedding, parent_embeddings.T).flatten()
                    parent_scores = similarities.tolist()
            
            # Step 3: Average parent scores to prevent bias
            if parent_scores:
                parent_boost = sum(parent_scores) * boost_weight / len(parent_scores)
            else:
                parent_boost = 0.0
            
            # Step 4: Calculate final score
            final_score = direct_score + parent_boost
            
            enhanced_results.append({
                'concept_id': concept_id,
                'concept_name': concept_name,
                'direct_score': direct_score,
                'parent_boost': parent_boost,
                'parent_count': len(parent_scores),
                'final_score': final_score
            })
        
        # Step 5: Sort and return top results
        enhanced_df = pd.DataFrame(enhanced_results)
        enhanced_df = enhanced_df.sort_values('final_score', ascending=False).head(top_k)
        
        logger.log(f"Level 1 parent hierarchical matching returned {len(enhanced_df)} results")
        return enhanced_df.reset_index(drop=True)


def load_hierarchical_matcher(data_folder: str = "data/omop_feather") -> HierarchicalMatcher:
    """
    Load a HierarchicalMatcher with OMOP data
    
    Args:
        data_folder: Path to folder containing concept_ancestor.feather and concept.feather
        
    Returns:
        Initialized HierarchicalMatcher instance
    """
    import os
    data_folder = "data/omop_feather"
    
    concept_ancestor = pd.read_feather(os.path.join(data_folder, 'concept_ancestor.feather'))
    concept_table = pd.read_feather(os.path.join(data_folder, 'concept.feather'))
    
    logger.log(f"Loaded {len(concept_ancestor)} ancestor relationships and {len(concept_table)} concepts")
    return HierarchicalMatcher(concept_ancestor, concept_table)


# Example usage and testing functions
def test_level1_parent_hierarchical_matching_with_model2():
    """Test hierarchical matching with Model 2 (Matching + Relation trained)"""
    from modules.ModelFunctions import auto_load_model
    from modules.FaissDB import build_index, is_initialized
    
    logger.log("Starting level 1 parent hierarchical matching test with Model 2")
    
    # Use the matching + relation trained model
    model, tokenizer, train_config = auto_load_model("output/finetune/2025-07-30_13-33-46")
    logger.log("Loaded Model 2 (Matching + Relation trained)")
    
    matcher = load_hierarchical_matcher()
    
    # Load standard concepts (assuming this file exists)
    try:
        std_concepts = pd.read_feather('data/matching/std_condition_concept.feather')
    except:
        # Fallback: use a subset of concept_table
        std_concepts = matcher.concept_table[['concept_id', 'concept_name']].head(10000)
    
    logger.log(f"Loaded {len(std_concepts)} standard concepts")
    
    # Build index for standard concepts
    repos_name = 'std_concepts_model2'
    if not is_initialized(repos=repos_name):
        build_index(
            model, 
            std_concepts['concept_id'].tolist(),
            std_concepts['concept_name'].tolist(),
            repos=repos_name
        )
        logger.log(f"Built index for {len(std_concepts)} concepts with Model 2")
    
    # Test cases - medical concepts with hierarchical relationships
    test_cases = [
        "prostate with metastasis, palliative treatment was given",
        "heart attack with chest pain and shortness of breath",
        "diabetes with complications and neuropathy",
        "chronic obstructive pulmonary disease with exacerbation",
        "malignant neoplasm of lung with pleural effusion"
    ]
    
    for test_input in test_cases:
        logger.log(f"\n{'='*60}")
        logger.log(f"Testing with Model 2: {test_input}")
        logger.log(f"{'='*60}")
        
        # Run the algorithm with Model 2
        results = matcher.hierarchical_matching_level1_parents(
            model=model,
            input_text=test_input,
            std_concepts=std_concepts,
            top_k=5,
            top_k_direct=20,
            boost_weight=0.5,  # May need adjustment with Model 2
            repos=repos_name
        )
        
        if not results.empty:
            print(f"\nModel 2 Results for: {test_input}")
            print("-" * 80)
            for _, row in results.iterrows():
                print(f"Concept: {row['concept_name']}")
                print(f"  Direct Score: {row['direct_score']:.3f}")
                print(f"  Parent Boost: {row['parent_boost']:.3f}")
                print(f"  Parent Count: {row['parent_count']}")
                print(f"  Final Score: {row['final_score']:.3f}")
                print("---")
        else:
            print(f"No results found for: {test_input}")
    
    return results

def test_level1_parent_hierarchical_matching():
    """Test the new level 1 parent hierarchical matching implementation"""
    from modules.ModelFunctions import get_ST_model
    from modules.FaissDB import build_index, is_initialized
    
    logger.log("Starting level 1 parent hierarchical matching test")
    
    model, _ = get_ST_model() 
    matcher = load_hierarchical_matcher()
    
    # Load standard concepts (assuming this file exists)
    try:
        std_concepts = pd.read_feather('data/matching/std_condition_concept.feather')
    except:
        # Fallback: use a subset of concept_table
        std_concepts = matcher.concept_table[['concept_id', 'concept_name']].head(10000)
    
    logger.log(f"Loaded {len(std_concepts)} standard concepts")
    
    # Build index for standard concepts
    repos_name = 'std_concepts_test'
    if not is_initialized(repos=repos_name):
        build_index(
            model, 
            std_concepts['concept_id'].tolist(),
            std_concepts['concept_name'].tolist(),
            repos=repos_name
        )
        logger.log(f"Built index for {len(std_concepts)} concepts")
    
    # Test cases
    test_cases = [
        "prostate with metastasis, palliative treatment was given",
        "heart attack with chest pain",
        "diabetes with high blood sugar"
    ]
    
    for test_input in test_cases:
        logger.log(f"\n{'='*50}")
        logger.log(f"Testing with input: {test_input}")
        logger.log(f"{'='*50}")
        
        # Run the algorithm
        results = matcher.hierarchical_matching_level1_parents(
            model=model,
            input_text=test_input,
            std_concepts=std_concepts,
            top_k=5,
            top_k_direct=20,
            repos=repos_name
        )
        
        if not results.empty:
            print(f"\nResults for: {test_input}")
            print("-" * 80)
            for _, row in results.iterrows():
                print(f"Concept: {row['concept_name']}")
                print(f"  Direct Score: {row['direct_score']:.3f}")
                print(f"  Parent Boost: {row['parent_boost']:.3f}")
                print(f"  Parent Count: {row['parent_count']}")
                print(f"  Final Score: {row['final_score']:.3f}")
                print("---")
        else:
            print(f"No results found for: {test_input}")
    
    return results


if __name__ == "__main__":
    # Run test with Model 2 (Matching + Relation trained)
    test_results = test_level1_parent_hierarchical_matching_with_model2()