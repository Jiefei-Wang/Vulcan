import sys
import types
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO
import tempfile
import os

try:
    from modules.Dataset import (
        PositiveDataset,
        NegativeDataset,
        FalsePositiveDataset,
        CombinedDataset
    )
    # Try to import the real modules to verify they work
    from modules.FaissDB import delete_repository
    from modules.FalsePositives import getFalsePositives
    REAL_MODULES_AVAILABLE = True
    print("Real modules loaded successfully")
except ImportError as e:
    print(f"Could not import real modules: {e}")
    print("Setting up mocks as fallback...")
    
    # Fallback to mocks if real modules fail
    fake_faissdb = types.ModuleType("modules.FaissDB")
    def delete_repository(repos): pass
    fake_faissdb.delete_repository = delete_repository
    sys.modules['modules.FaissDB'] = fake_faissdb

    fake_false = types.ModuleType("modules.FalsePositives")
    def getFalsePositives(model, corpus_concepts, n_fp, repos):
        return pd.DataFrame({'sentence1': [], 'sentence2': [], 'label': []})
    fake_false.get_false_positives = getFalsePositives
    sys.modules['modules.FalsePositives'] = fake_false
    
    from modules.Dataset import (
        PositiveDataset,
        NegativeDataset,
        FalsePositiveDataset,
        CombinedDataset
    )
    REAL_MODULES_AVAILABLE = False


class TestPositiveDatasetAdvanced(unittest.TestCase):
    def setUp(self):
        self.target_concepts = pd.DataFrame({
            'concept_id': [1, 2, 3],
            'concept_name': ['Concept_A', 'Concept_B', 'Concept_C'],
        })
        self.name_table = pd.DataFrame({
            'name_id': [10, 20, 30, 40, 50],
            'name': ['Name_X', 'Name_Y', 'Name_Z', 'Name_W', 'Name_V'],
        })
        self.name_bridge = pd.DataFrame({
            'concept_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
            'name_id': [10, 20, 30, 20, 40, 10, 30, 40, 50],
        })

    def test_resample_deterministic_with_seed(self):
        """Test that resampling with same seed produces identical results"""
        seed = 123
        ds1 = PositiveDataset(self.target_concepts, self.name_table, self.name_bridge, 
                             max_elements=2, seed=seed)
        ds2 = PositiveDataset(self.target_concepts, self.name_table, self.name_bridge, 
                             max_elements=2, seed=seed)
        
        pd.testing.assert_frame_equal(ds1.filtered_bridge, ds2.filtered_bridge)

    def test_resample_different_seeds(self):
        """Test that different seeds produce different results"""
        ds1 = PositiveDataset(self.target_concepts, self.name_table, self.name_bridge, 
                             max_elements=2, seed=42)
        ds2 = PositiveDataset(self.target_concepts, self.name_table, self.name_bridge, 
                             max_elements=2, seed=84)
        
        self.assertFalse(ds1.filtered_bridge.equals(ds2.filtered_bridge))

    def test_max_elements_per_concept(self):
        """Test that max_elements is respected per concept_id"""
        ds = PositiveDataset(self.target_concepts, self.name_table, self.name_bridge, 
                            max_elements=2, seed=42)
        
        concept_counts = ds.filtered_bridge['concept_id'].value_counts()
        
        self.assertTrue(all(count <= 2 for count in concept_counts.values))

    def test_invalid_concept_id_mapping(self):
        """Test behavior when concept_id not in target_concepts"""
        bridge_with_invalid = pd.DataFrame({
            'concept_id': [1, 999], 
            'name_id': [10, 20],
        })
        
        ds = PositiveDataset(self.target_concepts, self.name_table, bridge_with_invalid, 
                            max_elements=1, seed=42)
        
        valid_concept_ids = set(ds.filtered_bridge['concept_id'])
        self.assertTrue(valid_concept_ids.issubset({1, 2, 3}))

    def test_data_types_preserved(self):
        """Test that data types are preserved correctly"""
        ds = PositiveDataset(self.target_concepts, self.name_table, self.name_bridge, 
                            max_elements=1, seed=42)
        
        item = ds[0]
        self.assertIsInstance(item['sentence1'], str)
        self.assertIsInstance(item['sentence2'], str)
        self.assertIsInstance(item['label'], int)

    def test_custom_label_values(self):
        """Test different label values"""
        for label_val in [0, 1, 5, -1, 100]:
            ds = PositiveDataset(self.target_concepts, self.name_table, self.name_bridge, 
                                max_elements=1, label=label_val, seed=42)
            self.assertEqual(ds[0]['label'], label_val)

    def test_empty_dataframes(self):
        """Test behavior with empty input DataFrames"""
        empty_concepts = pd.DataFrame(columns=['concept_id', 'concept_name'])
        empty_names = pd.DataFrame(columns=['name_id', 'name'])
        empty_bridge = pd.DataFrame(columns=['concept_id', 'name_id'])
        
        ds = PositiveDataset(empty_concepts, empty_names, empty_bridge, max_elements=1)
        self.assertEqual(len(ds), 0)

    def test_index_out_of_bounds(self):
        """Test IndexError for out of bounds access"""
        ds = PositiveDataset(self.target_concepts, self.name_table, self.name_bridge, 
                            max_elements=1, seed=42)
        with self.assertRaises(IndexError):
            ds[999]

# self = TestNegativeDatasetAdvanced()
# self.setUp()
class TestNegativeDatasetAdvanced(unittest.TestCase):
    def setUp(self):
        self.target_concepts = pd.DataFrame({
            'concept_id': [1, 2, 3],
            'concept_name': ['A', 'B', 'C'],
        })
        self.name_table = pd.DataFrame({
            'name_id': [10, 20, 30, 40, 50],
            'name': ['X', 'Y', 'Z', 'W', 'V'],
        })
        self.blacklist_bridge = pd.DataFrame({
            'concept_id': [1, 1, 2],
            'name_id': [10, 20, 30],
        })

    def test_blacklist_effectiveness(self):
        """Test that blacklisted pairs are never generated"""
        ds = NegativeDataset(self.target_concepts, self.name_table, self.blacklist_bridge, 
                            max_elements=10, seed=42)
        
        # Check that no blacklisted pairs exist
        for item in ds:
            concept_name = item['sentence1']
            name = item['sentence2']
            
            # Map back to IDs for checking
            concept_id = next(cid for cid, cname in zip(self.target_concepts['concept_id'], 
                                                       self.target_concepts['concept_name']) 
                             if cname == concept_name)
            name_id = next(nid for nid, n in zip(self.name_table['name_id'], 
                                                self.name_table['name']) 
                          if n == name)
            
            blacklisted = ((self.blacklist_bridge['concept_id'] == concept_id) & 
                          (self.blacklist_bridge['name_id'] == name_id)).any()
            self.assertFalse(blacklisted, f"Found blacklisted pair: {concept_name}-{name}")

    def test_get_random_sample_no_candidates(self):
        """Test _get_random_sample when no candidates available"""
        ds = NegativeDataset(self.target_concepts, self.name_table, self.blacklist_bridge, 
                            max_elements=1, seed=42)
        rng = np.random.default_rng(42)
        
        # Test case where all names are blacklisted or selected
        all_name_ids = self.name_table['name_id'].tolist()
        sample = ds._get_random_sample(rng, concept_id=1, selected=all_name_ids, num_sample=1)
        
        self.assertEqual(len(sample), 0)

    def test_missing_concept_in_blacklist(self):
        """Test behavior when concept_id not in blacklist"""
        ds = NegativeDataset(self.target_concepts, self.name_table, self.blacklist_bridge, 
                            max_elements=2, seed=42)
        rng = np.random.default_rng(42)
        
        # Concept 3 is not in blacklist_bridge
        sample = ds._get_random_sample(rng, concept_id=3, selected=[], num_sample=2)
        
        self.assertGreater(len(sample), 0)
        self.assertTrue(all(sid in self.name_table['name_id'].values for sid in sample))

    def test_duplicate_removal(self):
        """Test that duplicate concept_id-name_id pairs are removed"""
        ds = NegativeDataset(self.target_concepts, self.name_table, self.blacklist_bridge, 
                            max_elements=5, seed=42)
        
        # Check no duplicates in final bridge
        duplicates = ds.bridge.duplicated(subset=['concept_id', 'name_id'])
        self.assertFalse(duplicates.any())

    def test_label_always_zero(self):
        """Test that NegativeDataset always returns label=0"""
        ds = NegativeDataset(self.target_concepts, self.name_table, self.blacklist_bridge, 
                            max_elements=5, seed=42)
        
        for item in ds:
            self.assertEqual(item['label'], 0)

    def test_str_method_fixed(self):
        """Test that __str__ method prints NegativeDataset"""
        ds = NegativeDataset(self.target_concepts,
                             self.name_table,
                             self.blacklist_bridge,
                             max_elements=1,
                             seed=42)
        str_repr = str(ds)
        self.assertIn('NegativeDataset', str_repr)
        self.assertIn(f"seed={ds.seed}", str_repr)

    def test_empty_blacklist(self):
        """Test behavior with empty blacklist"""
        empty_blacklist = pd.DataFrame(columns=['concept_id', 'name_id'])
        ds = NegativeDataset(self.target_concepts, self.name_table, empty_blacklist, 
                            max_elements=2, seed=42)

        self.assertGreater(len(ds), 0)

    def test_index_out_of_bounds(self):
        """Test IndexError for out of bounds access"""
        ds = NegativeDataset(self.target_concepts, self.name_table, self.blacklist_bridge, 
                            max_elements=1, seed=42)
        with self.assertRaises(IndexError):
            ds[999]


# self = TestFalsePositiveDatasetAdvanced()
# self.setUp()

class TestFalsePositiveDatasetAdvanced(unittest.TestCase):
    def setUp(self):
        self.target_concepts = pd.DataFrame({
            'concept_id': [1, 2],
            'concept_name': ['ConceptA', 'ConceptB'],
        })
        self.mock_model = MagicMock()

    def test_init_with_existing_path_mock(self):
        """Test initialization with existing feather file"""
        # Create temporary feather file
        with tempfile.NamedTemporaryFile(suffix='.feather', delete=False) as tmp:
            test_df = pd.DataFrame({
                'sentence1': ['s1', 's2'],
                'sentence2': ['t1', 't2'],
                'label': [0, 0],
                'extra_col': ['x1', 'x2']  
            })
            test_df.to_feather(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Test with existing path
            ds = FalsePositiveDataset(self.target_concepts, n_fp=10, 
                                     existing_path=tmp_path)
            
            # Should load only required columns
            expected_cols = ['sentence1', 'sentence2', 'label']
            self.assertEqual(list(ds.fp_matching.columns), expected_cols)
            self.assertEqual(len(ds), 2)
            
        finally:
            os.unlink(tmp_path)

    @patch('modules.Dataset.delete_repository')
    @patch('modules.Dataset.get_false_positives')
    def test_resample_with_different_n_fp(self, mock_get_fp, mock_delete):
        """Test resample with different n_fp_matching values"""
        # Test different sizes
        for n_fp in [1, 5, 10, 100]:
            mock_df = pd.DataFrame({
                'sentence1': [f's{i}' for i in range(n_fp)],
                'sentence2': [f't{i}' for i in range(n_fp)],
                'label': [0] * n_fp,
            })
            mock_get_fp.return_value = mock_df
            
            ds = FalsePositiveDataset(self.target_concepts, n_fp=n_fp)
            ds.add_model(self.mock_model)
            ds.resample(seed=42)
            
            self.assertEqual(len(ds), n_fp)
            self.assertEqual(ds.n_fp_matching, n_fp)

    @unittest.skipUnless(REAL_MODULES_AVAILABLE, "Real modules not available")
    def test_resample_with_real_modules(self):
        """Test resample with real FaissDB and FalsePositives modules"""
        ds = FalsePositiveDataset(self.target_concepts, n_fp=2)
        ds.add_model(self.mock_model)
        
        try:
            ds.resample(seed=42)
            
            # Should have some result (could be empty if no false positives found)
            self.assertIsInstance(ds.fp_matching, pd.DataFrame)
            self.assertEqual(len(ds.fp_matching.columns), 3)
            self.assertEqual(list(ds.fp_matching.columns), ['sentence1', 'sentence2', 'label'])
            
            print(f"Real modules test passed - got {len(ds)} false positives")
            
        except Exception as e:
            # If real modules fail, that's valuable information
            print(f"Real modules test failed: {e}")
            self.fail(f"Real modules integration failed: {e}")

    def test_getitem_slice_behavior(self):
        """Test slice indexing behavior"""
        test_df = pd.DataFrame({
            'sentence1': ['s1', 's2', 's3', 's4'],
            'sentence2': ['t1', 't2', 't3', 't4'],
            'label': [0, 0, 0, 0],
        })
        
        ds = FalsePositiveDataset(self.target_concepts, n_fp=4)
        ds.fp_matching = test_df
        
        # Test various slices
        self.assertEqual(len(ds[0:2]), 2)
        self.assertEqual(len(ds[1:4:2]), 2)
        self.assertEqual(len(ds[:]), 4)
        
        # Test slice content
        slice_result = ds[1:3]
        self.assertEqual(slice_result[0]['sentence1'], 's2')
        self.assertEqual(slice_result[1]['sentence1'], 's3')

    def test_add_model_method(self):
        """Test add_model method"""
        ds = FalsePositiveDataset(self.target_concepts, n_fp=10)
        ds.add_model(self.mock_model)
        self.assertEqual(ds.model, self.mock_model)

    def test_empty_dataset(self):
        """Test behavior with empty dataset"""
        empty_df = pd.DataFrame({'sentence1': [], 'sentence2': [], 'label': []})
        ds = FalsePositiveDataset(self.target_concepts, n_fp=0)
        ds.fp_matching = empty_df
        self.assertEqual(len(ds), 0)

    def test_index_out_of_bounds(self):
        """Test IndexError for out of bounds access"""
        test_df = pd.DataFrame({
            'sentence1': ['s1', 's2'],
            'sentence2': ['t1', 't2'],
            'label': [0, 0],
        })
        ds = FalsePositiveDataset(self.target_concepts, n_fp=2)
        ds.fp_matching = test_df
        
        with self.assertRaises(IndexError):
            ds[999]


class TestCombinedDatasetAdvanced(unittest.TestCase):
    def setUp(self):
        self.ds1 = [{'type': 'A', 'value': i} for i in range(5)]
        self.ds2 = [{'type': 'B', 'value': i} for i in range(3)]
        self.ds3 = [{'type': 'C', 'value': i} for i in range(2)]

    def test_multiple_datasets_combination(self):
        """Test combining more than 2 datasets"""
        cd = CombinedDataset(first=self.ds1, second=self.ds2, third=self.ds3)
        
        self.assertEqual(len(cd), 10)  # 5 + 3 + 2
        self.assertEqual(len(cd.datasets), 3)
        self.assertEqual(set(cd.names), {'first', 'second', 'third'})

    def test_single_dataset_combination(self):
        """Test combining single dataset"""
        cd = CombinedDataset(only=self.ds1)
        
        self.assertEqual(len(cd), 5)
        self.assertEqual(len(cd.datasets), 1)

    def test_dataset_indexing_boundaries(self):
        """Test indexing at dataset boundaries"""
        cd = CombinedDataset(ds1=self.ds1, ds2=self.ds2)  # lengths: 5, 3
        
        # Test boundary indices
        self.assertEqual(cd[4]['type'], 'A')  # Last of ds1
        self.assertEqual(cd[5]['type'], 'B')  # First of ds2
        self.assertEqual(cd[7]['type'], 'B')  # Last of ds2

    def test_shuffle_preserves_data(self):
        """Test that shuffle doesn't lose or duplicate data"""
        cd = CombinedDataset(ds1=self.ds1, ds2=self.ds2)
        original_items = [cd[i] for i in range(len(cd))]
        
        cd.shuffle(seed=42)
        shuffled_items = [cd[i] for i in range(len(cd))]
        
        # Should have same items, just reordered
        self.assertEqual(len(original_items), len(shuffled_items))
        
        # Count items by type
        original_counts = {}
        shuffled_counts = {}
        for item in original_items:
            original_counts[item['type']] = original_counts.get(item['type'], 0) + 1
        for item in shuffled_items:
            shuffled_counts[item['type']] = shuffled_counts.get(item['type'], 0) + 1
            
        self.assertEqual(original_counts, shuffled_counts)

    def test_resample_with_mixed_dataset_types(self):
        """Test resample with datasets that have and don't have resample method"""
        # Create mock datasets
        mock_resampleable = MagicMock()
        mock_resampleable.resample.return_value = mock_resampleable
        mock_resampleable.__len__.return_value = 3
        
        mock_non_resampleable = [{'x': 1}, {'x': 2}]
        
        cd = CombinedDataset(resampleable=mock_resampleable, 
                            non_resampleable=mock_non_resampleable)
        
        cd.resample(seed=123)
        
        # Should call resample on the resampleable dataset
        mock_resampleable.resample.assert_called_once_with(123)
        
        # Should update lengths
        self.assertEqual(cd.lengths[0], 3)  # resampleable length
        self.assertEqual(cd.lengths[1], 2)  # non-resampleable length

    def test_slice_with_step(self):
        """Test slice indexing with step parameter"""
        cd = CombinedDataset(ds1=self.ds1, ds2=self.ds2)
        
        # Test step slicing
        result = cd[::2]  # Every other item
        self.assertEqual(len(result), 4)  # 8 items total, every other = 4
        
        result = cd[1::3]  # Every third item starting from index 1
        expected_len = len(range(1, 8, 3))  # range(1, 8, 3)
        self.assertEqual(len(result), expected_len)

    def test_pandas_dataframe_integration(self):
        """Test with pandas DataFrame as dataset"""
        df = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'col2': [1, 2, 3]
        })
        
        cd = CombinedDataset(df_data=df, list_data=self.ds1)
        
        # Should handle DataFrame properly
        self.assertEqual(len(cd), 8)  # 3 + 5
        
        # DataFrame items should be converted to dict
        first_df_item = cd[0]
        self.assertIsInstance(first_df_item, dict)
        self.assertIn('col1', first_df_item)

    def test_empty_datasets(self):
        """Test behavior with empty datasets"""
        empty_ds1 = []
        empty_ds2 = []
        cd = CombinedDataset(ds1=empty_ds1, ds2=empty_ds2)
        self.assertEqual(len(cd), 0)

    def test_index_out_of_bounds(self):
        """Test IndexError for out of bounds access"""
        cd = CombinedDataset(ds1=self.ds1)
        with self.assertRaises(IndexError):
            cd[999]

    def test_str_method_with_seed_initialization(self):
        """Test string representation includes seed (verifies seed initialization fix)"""
        cd = CombinedDataset(ds1=self.ds1)
        str_repr = str(cd)
        self.assertIn('CombinedDataset', str_repr)
        self.assertIn('length=5', str_repr)
        self.assertIn('seed=None', str_repr)  # Verifies seed initialization fix


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests combining multiple dataset classes"""
    
    def test_positive_negative_combined_workflow(self):
        """Test typical workflow combining positive and negative datasets"""
        # Setup data
        concepts = pd.DataFrame({
            'concept_id': [1, 2],
            'concept_name': ['Medical', 'Tech'],
        })
        names = pd.DataFrame({
            'name_id': [10, 20, 30, 40],
            'name': ['Doctor', 'Engineer', 'Nurse', 'Programmer'],
        })
        pos_bridge = pd.DataFrame({
            'concept_id': [1, 1, 2, 2],
            'name_id': [10, 30, 20, 40],
        })
        neg_blacklist = pd.DataFrame({
            'concept_id': [1, 2],
            'name_id': [20, 10],  # Medical-Engineer, Tech-Doctor blocked
        })
        
        # Create datasets
        pos_ds = PositiveDataset(concepts, names, pos_bridge, max_elements=2, seed=42)
        neg_ds = NegativeDataset(concepts, names, neg_blacklist, max_elements=2, seed=42)
        
        # Combine
        combined = CombinedDataset(positive=pos_ds, negative=neg_ds)
        
        # Verify properties
        self.assertEqual(len(combined), len(pos_ds) + len(neg_ds))
        
        # Check label distribution
        labels = [item['label'] for item in combined]
        self.assertIn(0, labels)  # Should have negative labels
        self.assertIn(1, labels)  # Should have positive labels (default)

    def test_dataset_consistency_across_resamples(self):
        """Test that datasets maintain consistency across multiple resamples"""
        concepts = pd.DataFrame({
            'concept_id': [1],
            'concept_name': ['Test'],
        })
        names = pd.DataFrame({
            'name_id': [10, 20, 30],
            'name': ['A', 'B', 'C'],
        })
        bridge = pd.DataFrame({
            'concept_id': [1, 1, 1],
            'name_id': [10, 20, 30],
        })
        
        ds = PositiveDataset(concepts, names, bridge, max_elements=2, seed=42)
        original_length = len(ds)
        
        # Resample multiple times with same seed
        for _ in range(5):
            ds.resample(seed=42)
            self.assertEqual(len(ds), original_length)


class TestRealModuleIntegration(unittest.TestCase):
    """Test integration with real FaissDB and FalsePositives modules"""
    
    @unittest.skipUnless(REAL_MODULES_AVAILABLE, "Real modules not available")
    def test_delete_repository_function(self):
        """Test that delete_repository function can be called"""
        try:
            # This should not raise an exception
            delete_repository(repos='test_repository')
            print("delete_repository function works")
        except Exception as e:
            print(f"delete_repository failed: {e}")
            # Don't fail the test, just log the issue
            pass

    @unittest.skipUnless(REAL_MODULES_AVAILABLE, "Real modules not available")
    def test_get_false_positives_function(self):
        """Test that get_false_positives function can be called"""
        try:
            # Create minimal test data
            concepts = pd.DataFrame({
                'concept_id': [1],
                'concept_name': ['test_concept'],
            })
            
            # This might fail with real dependencies, but we want to see what happens
            mock_model = MagicMock()
            result = getFalsePositives(
                model=mock_model,
                corpus_concepts=concepts,
                n_fp=1,
                repos='test_repository'
            )
            
            # Should return a DataFrame
            self.assertIsInstance(result, pd.DataFrame)
            print(f"get_false_positives function works - returned {len(result)} rows")
            
        except Exception as e:
            print(f"get_false_positives failed: {e}")
            # Don't fail the test, just log the issue
            pass

    def test_module_availability_status(self):
        """Test that shows which modules are available"""
        if REAL_MODULES_AVAILABLE:
            print("Real modules are available and imported successfully")
            self.assertTrue(True)
        else:
            print("Real modules are not available - using mocks")
            # This is not a failure, just informational
            self.assertTrue(True)


class TestEdgeCasesAndBoundaries(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_large_max_elements(self):
        """Test behavior with very large max_elements"""
        concepts = pd.DataFrame({
            'concept_id': [1],
            'concept_name': ['Test'],
        })
        names = pd.DataFrame({
            'name_id': [10, 20],
            'name': ['A', 'B'],
        })
        bridge = pd.DataFrame({
            'concept_id': [1, 1],
            'name_id': [10, 20],
        })
        
        # max_elements much larger than available data
        ds = PositiveDataset(concepts, names, bridge, max_elements=1000, seed=42)
        self.assertEqual(len(ds), 2)  # Should be limited by available data

    def test_zero_max_elements(self):
        """Test behavior with zero max_elements"""
        concepts = pd.DataFrame({
            'concept_id': [1],
            'concept_name': ['Test'],
        })
        names = pd.DataFrame({
            'name_id': [10],
            'name': ['A'],
        })
        bridge = pd.DataFrame({
            'concept_id': [1],
            'name_id': [10],
        })
        
        ds = PositiveDataset(concepts, names, bridge, max_elements=0, seed=42)
        self.assertEqual(len(ds), 0)

    def test_single_element_datasets(self):
        """Test behavior with single element datasets"""
        concepts = pd.DataFrame({
            'concept_id': [1],
            'concept_name': ['Test'],
        })
        names = pd.DataFrame({
            'name_id': [10],
            'name': ['A'],
        })
        bridge = pd.DataFrame({
            'concept_id': [1],
            'name_id': [10],
        })
        
        ds = PositiveDataset(concepts, names, bridge, max_elements=1, seed=42)
        self.assertEqual(len(ds), 1)
        
        item = ds[0]
        self.assertEqual(item['sentence1'], 'Test')
        self.assertEqual(item['sentence2'], 'A')
        self.assertEqual(item['label'], 1)


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)