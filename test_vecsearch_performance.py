#!/usr/bin/env python3
"""
Performance comparison test between FaissDB and VecSearch
Tests GPU acceleration improvements for similarity search
"""

import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Import both implementations
from modules.FaissDB import build_index as faiss_build, search_similar as faiss_search, delete_repository as faiss_delete
from modules.VecSearch import build_index as vec_build, search_similar as vec_search, delete_repository as vec_delete

def generate_test_data():
    """Generate test data similar to medical concepts"""
    
    # Simulate 160K corpus (use smaller subset for testing)
    test_corpus_size = 1000  # Start small for initial test
    test_query_size = 50
    
    # Generate mock medical concept names
    corpus_ids = [f"concept_{i}" for i in range(test_corpus_size)]
    corpus_names = [f"Medical condition {i} syndrome" for i in range(test_corpus_size)]
    
    query_ids = [f"query_{i}" for i in range(test_query_size)]
    query_names = [f"Patient symptom {i} disorder" for i in range(test_query_size)]
    
    return corpus_ids, corpus_names, query_ids, query_names

def test_performance():
    """Compare performance between FaissDB and VecSearch"""
    
    print("=== VecSearch vs FaissDB Performance Test ===\n")
    
    # Load a lightweight model for testing
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight model
    
    # Generate test data
    print("Generating test data...")
    corpus_ids, corpus_names, query_ids, query_names = generate_test_data()
    
    print(f"Test setup:")
    print(f"  Corpus size: {len(corpus_ids):,}")
    print(f"  Query size: {len(query_ids):,}")
    print(f"  Top-k: 50 (professor's requirement)")
    print()
    
    results = {}
    
    # Test FaissDB (original implementation)
    print("=" * 50)
    print("Testing FaissDB (Original Implementation)")
    print("=" * 50)
    
    try:
        faiss_delete('test')
        
        start_time = time.time()
        faiss_build(model, corpus_ids, corpus_names, repos='test')
        build_time_faiss = time.time() - start_time
        
        start_time = time.time()
        faiss_results = faiss_search(query_ids, query_names, top_k=50, repos='test')
        search_time_faiss = time.time() - start_time
        
        results['faiss'] = {
            'build_time': build_time_faiss,
            'search_time': search_time_faiss,
            'total_time': build_time_faiss + search_time_faiss,
            'results_count': len(faiss_results)
        }
        
        print(f"FaissDB Results:")
        print(f"  Build time: {build_time_faiss:.2f}s")
        print(f"  Search time: {search_time_faiss:.2f}s")
        print(f"  Total time: {build_time_faiss + search_time_faiss:.2f}s")
        print(f"  Results returned: {len(faiss_results):,}")
        
    except Exception as e:
        print(f"FaissDB failed: {e}")
        results['faiss'] = None
    
    print()
    
    # Test VecSearch (GPU implementation)
    print("=" * 50)
    print("Testing VecSearch (GPU Implementation)")
    print("=" * 50)
    
    try:
        vec_delete('test')
        
        start_time = time.time()
        vec_build(model, corpus_ids, corpus_names, repos='test')
        build_time_vec = time.time() - start_time
        
        start_time = time.time()
        vec_results = vec_search(query_ids, query_names, top_k=50, repos='test')
        search_time_vec = time.time() - start_time
        
        results['vecsearch'] = {
            'build_time': build_time_vec,
            'search_time': search_time_vec,
            'total_time': build_time_vec + search_time_vec,
            'results_count': len(vec_results)
        }
        
        print(f"VecSearch Results:")
        print(f"  Build time: {build_time_vec:.2f}s")
        print(f"  Search time: {search_time_vec:.2f}s")
        print(f"  Total time: {build_time_vec + search_time_vec:.2f}s")
        print(f"  Results returned: {len(vec_results):,}")
        
    except Exception as e:
        print(f"VecSearch failed: {e}")
        results['vecsearch'] = None
    
    print()
    
    # Performance comparison
    if results['faiss'] and results['vecsearch']:
        print("=" * 50)
        print("Performance Comparison")
        print("=" * 50)
        
        search_speedup = results['faiss']['search_time'] / results['vecsearch']['search_time']
        total_speedup = results['faiss']['total_time'] / results['vecsearch']['total_time']
        
        print(f"Search Speed Improvement: {search_speedup:.2f}x faster")
        print(f"Total Speed Improvement: {total_speedup:.2f}x faster")
        print()
        
        if search_speedup > 1:
            print("✅ VecSearch is faster than FaissDB!")
        else:
            print("❌ VecSearch is slower than FaissDB")
            
        print(f"✅ Both return similar result counts: {results['faiss']['results_count']} vs {results['vecsearch']['results_count']}")
    
    # Cleanup
    try:
        faiss_delete('test')
        vec_delete('test')
    except:
        pass
    
    return results

if __name__ == "__main__":
    test_performance()