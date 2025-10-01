import pandas as pd
import numpy as np
import tempfile
import os

from modules.Dataset import (
    PositiveDataset,
    NegativeDataset,
    FalsePositiveDataset,
    CombinedDataset
)
# Try to import the real modules to verify they work
from modules.FaissDB import delete_repository
from modules.FalsePositives import getFalsePositives


print("Starting PositiveDataset tests...")

# Test data setup
target_concepts = pd.DataFrame({
    'concept_id': [1, 2, 3],
    'concept_name': ['Concept_A', 'Concept_B', 'Concept_C'],
})
name_table = pd.DataFrame({
    'name_id': [10, 20, 30, 40, 50],
    'name': ['Name_X', 'Name_Y', 'Name_Z', 'Name_W', 'Name_V'],
})
name_bridge = pd.DataFrame({
    'concept_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
    'name_id': [10, 20, 30, 20, 40, 10, 30, 40, 50],
})

# Test PositiveDataset - resample deterministic with seed
print("Testing PositiveDataset resample deterministic with seed...")
seed = 123
ds1 = PositiveDataset(target_concepts, name_table, name_bridge, max_elements=2, seed=seed)
ds2 = PositiveDataset(target_concepts, name_table, name_bridge, max_elements=2, seed=seed)
pd.testing.assert_frame_equal(ds1.filtered_bridge, ds2.filtered_bridge)
print("- Same seed produces identical results")

# Test PositiveDataset - different seeds produce different results
print("Testing PositiveDataset with different seeds...")
ds1 = PositiveDataset(target_concepts, name_table, name_bridge, max_elements=2, seed=42)
ds2 = PositiveDataset(target_concepts, name_table, name_bridge, max_elements=2, seed=84)
assert not ds1.filtered_bridge.equals(ds2.filtered_bridge), "Different seeds should produce different results"
print("- Different seeds produce different results")

# Test PositiveDataset - max_elements per concept
print("Testing PositiveDataset max_elements per concept...")
ds = PositiveDataset(target_concepts, name_table, name_bridge, max_elements=2, seed=42)
concept_counts = ds.filtered_bridge['concept_id'].value_counts()
assert all(count <= 2 for count in concept_counts.values), "max_elements should be respected per concept_id"
print("- max_elements is respected per concept_id")

# Test PositiveDataset - invalid concept_id mapping
print("Testing PositiveDataset with invalid concept_id...")
bridge_with_invalid = pd.DataFrame({
    'concept_id': [1, 999], 
    'name_id': [10, 20],
})
ds = PositiveDataset(target_concepts, name_table, bridge_with_invalid, max_elements=1, seed=42)
valid_concept_ids = set(ds.filtered_bridge['concept_id'])
assert valid_concept_ids.issubset({1, 2, 3}), "Invalid concept_ids should be filtered out"
print("- Invalid concept_ids are filtered out")

# Test PositiveDataset - data types preserved
print("Testing PositiveDataset data types...")
ds = PositiveDataset(target_concepts, name_table, name_bridge, max_elements=1, seed=42)
if len(ds) > 0:
    item = ds[0]
    assert isinstance(item['sentence1'], str), "sentence1 should be string"
    assert isinstance(item['sentence2'], str), "sentence2 should be string"
    assert isinstance(item['label'], int), "label should be int"
    print("- Data types are preserved correctly")

# Test PositiveDataset - custom label values
print("Testing PositiveDataset custom label values...")
for label_val in [0, 1, 5, -1, 100]:
    ds = PositiveDataset(target_concepts, name_table, name_bridge, max_elements=1, label=label_val, seed=42)
    if len(ds) > 0:
        assert ds[0]['label'] == label_val, f"Label should be {label_val}"
print("- Custom label values work correctly")

# Test PositiveDataset - empty dataframes
print("Testing PositiveDataset with empty dataframes...")
empty_concepts = pd.DataFrame(columns=['concept_id', 'concept_name'])
empty_names = pd.DataFrame(columns=['name_id', 'name'])
empty_bridge = pd.DataFrame(columns=['concept_id', 'name_id'])
ds = PositiveDataset(empty_concepts, empty_names, empty_bridge, max_elements=1)
assert len(ds) == 0, "Empty dataframes should result in empty dataset"
print("- Empty dataframes handled correctly")

# Test PositiveDataset - index out of bounds
print("Testing PositiveDataset index out of bounds...")
ds = PositiveDataset(target_concepts, name_table, name_bridge, max_elements=1, seed=42)
try:
    ds[999]
    assert False, "Should have raised IndexError"
except IndexError:
    print("- IndexError raised for out of bounds access")

print("\nStarting NegativeDataset tests...")

# Test data for NegativeDataset
target_concepts_neg = pd.DataFrame({
    'concept_id': [1, 2, 3],
    'concept_name': ['A', 'B', 'C'],
})
name_table_neg = pd.DataFrame({
    'name_id': [10, 20, 30, 40, 50],
    'name': ['X', 'Y', 'Z', 'W', 'V'],
})
blacklist_bridge = pd.DataFrame({
    'concept_id': [1, 1, 2],
    'name_id': [10, 20, 30],
})

# Test NegativeDataset - blacklist effectiveness
print("Testing NegativeDataset blacklist effectiveness...")
ds = NegativeDataset(target_concepts_neg, name_table_neg, blacklist_bridge, max_elements=10, seed=42)
for i in range(min(len(ds), 10)):  # Check up to 10 items
    item = ds[i]
    concept_name = item['sentence1']
    name = item['sentence2']
    
    # Map back to IDs for checking
    concept_id = next(cid for cid, cname in zip(target_concepts_neg['concept_id'], 
                                               target_concepts_neg['concept_name']) 
                     if cname == concept_name)
    name_id = next(nid for nid, n in zip(name_table_neg['name_id'], 
                                        name_table_neg['name']) 
                  if n == name)
    
    blacklisted = ((blacklist_bridge['concept_id'] == concept_id) & 
                  (blacklist_bridge['name_id'] == name_id)).any()
    assert not blacklisted, f"Found blacklisted pair: {concept_name}-{name}"
print("- Blacklisted pairs are never generated")

# Test NegativeDataset - label always zero
print("Testing NegativeDataset label always zero...")
ds = NegativeDataset(target_concepts_neg, name_table_neg, blacklist_bridge, max_elements=5, seed=42)
for i in range(min(len(ds), 5)):
    item = ds[i]
    assert item['label'] == 0, "NegativeDataset should always return label=0"
print("- NegativeDataset always returns label=0")

# Test NegativeDataset - empty blacklist
print("Testing NegativeDataset with empty blacklist...")
empty_blacklist = pd.DataFrame(columns=['concept_id', 'name_id'])
ds = NegativeDataset(target_concepts_neg, name_table_neg, empty_blacklist, max_elements=2, seed=42)
assert len(ds) > 0, "Should generate negative examples even with empty blacklist"
print("- Empty blacklist handled correctly")

print("\nStarting FalsePositiveDataset tests...")

# Test data for FalsePositiveDataset
target_concepts_fp = pd.DataFrame({
    'concept_id': [1, 2],
    'concept_name': ['ConceptA', 'ConceptB'],
})

# Test FalsePositiveDataset - init with existing path
print("Testing FalsePositiveDataset with existing feather file...")
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
    ds = FalsePositiveDataset(
        corpus_ids=target_concepts_fp['concept_id'],
        corpus_names=target_concepts_fp['concept_name'],
        n_fp=10, existing_path=tmp_path)
    expected_cols = ['sentence1', 'sentence2', 'label']
    assert list(ds.fp.columns) == expected_cols, "Should load only required columns"
    assert len(ds) == 2, "Should load correct number of rows"
    print("- Existing feather file loaded correctly")
finally:
    os.unlink(tmp_path)

# Test FalsePositiveDataset - empty dataset
print("Testing FalsePositiveDataset with empty dataset...")
empty_df = pd.DataFrame({'sentence1': [], 'sentence2': [], 'label': []})
ds = FalsePositiveDataset(
        corpus_ids=target_concepts_fp['concept_id'],
        corpus_names=target_concepts_fp['concept_name'],
        n_fp=0)
ds.fp = empty_df
assert len(ds) == 0, "Empty dataset should have length 0"
print("- Empty dataset handled correctly")

print("\nStarting CombinedDataset tests...")

# Test data for CombinedDataset
ds1 = [{'type': 'A', 'value': i} for i in range(5)]
ds2 = [{'type': 'B', 'value': i} for i in range(3)]
ds3 = [{'type': 'C', 'value': i} for i in range(2)]

# Test CombinedDataset - multiple datasets combination
print("Testing CombinedDataset with multiple datasets...")
cd = CombinedDataset(first=ds1, second=ds2, third=ds3)
assert len(cd) == 10, "Combined length should be 5 + 3 + 2 = 10"
assert len(cd.datasets) == 3, "Should have 3 datasets"
assert set(cd.names) == {'first', 'second', 'third'}, "Names should match"
print("- Multiple datasets combined correctly")

# Test CombinedDataset - single dataset combination
print("Testing CombinedDataset with single dataset...")
cd = CombinedDataset(only=ds1)
assert len(cd) == 5, "Single dataset length should be 5"
assert len(cd.datasets) == 1, "Should have 1 dataset"
print("- Single dataset handled correctly")

# Test CombinedDataset - dataset indexing boundaries
print("Testing CombinedDataset indexing boundaries...")
cd = CombinedDataset(ds1=ds1, ds2=ds2)  # lengths: 5, 3
assert cd[4]['type'] == 'A', "Index 4 should be last of ds1"
assert cd[5]['type'] == 'B', "Index 5 should be first of ds2"
assert cd[7]['type'] == 'B', "Index 7 should be last of ds2"
print("- Dataset boundary indexing works correctly")

# Test CombinedDataset - empty datasets
print("Testing CombinedDataset with empty datasets...")
empty_ds1 = []
empty_ds2 = []
cd = CombinedDataset(ds1=empty_ds1, ds2=empty_ds2)
assert len(cd) == 0, "Empty datasets should result in length 0"
print("- Empty datasets handled correctly")

print("\nStarting integration tests...")

# Test positive-negative combined workflow
print("Testing positive-negative combined workflow...")
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

pos_ds = PositiveDataset(concepts, names, pos_bridge, max_elements=2, seed=42)
neg_ds = NegativeDataset(concepts, names, neg_blacklist, max_elements=2, seed=42)
combined = CombinedDataset(positive=pos_ds, negative=neg_ds)

assert len(combined) == len(pos_ds) + len(neg_ds), "Combined length should be sum of components"
labels = [item['label'] for item in combined]
assert 0 in labels, "Should have negative labels"
assert 1 in labels, "Should have positive labels"
print("- Positive-negative workflow works correctly")

print("\nStarting edge case tests...")

# Test large max_elements
print("Testing large max_elements...")
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
ds = PositiveDataset(concepts, names, bridge, max_elements=1000, seed=42)
assert len(ds) == 2, "Should be limited by available data"
print("- Large max_elements handled correctly")

# Test zero max_elements
print("Testing zero max_elements...")
ds = PositiveDataset(concepts, names, bridge, max_elements=0, seed=42)
assert len(ds) == 0, "Zero max_elements should result in empty dataset"
print("- Zero max_elements handled correctly")

# Test single element datasets
print("Testing single element datasets...")
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
assert len(ds) == 1, "Single element dataset should have length 1"

item = ds[0]
assert item['sentence1'] == 'Test', "sentence1 should match concept_name"
assert item['sentence2'] == 'A', "sentence2 should match name"
assert item['label'] == 1, "label should be 1 for positive dataset"
print("- Single element dataset works correctly")


print("\nStarting comprehensive PositiveDataset tests...")

# Comprehensive test data
target_concepts_comp = pd.DataFrame({
    'concept_id': [101, 102, 103, 104],
    'concept_name': ['a', 'b', 'c', 'd']
})

name_table_comp = pd.DataFrame({
    'name_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'name': ['a1', 'b1', 'b2', 'c1', 'c2', 'c3', 'd1', 'd2', 'd3', 'd4']
})

name_bridge_comp = pd.DataFrame({
    'concept_id': [101, 102, 102, 103, 103, 103, 104, 104, 104, 104],
    'name_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
})

# Test PositiveDataset with max_elements=10 (no limit)
print("Testing PositiveDataset with no element limit...")
pos_ds = PositiveDataset(
    target_concepts=target_concepts_comp,
    name_table=name_table_comp,
    name_bridge=name_bridge_comp,
    max_elements=10
)

df = pd.DataFrame(iter(pos_ds))
# sentence2 should start with the letter in sentence1
assert all(df['sentence1'].str[0] == df['sentence2'].str[0]), "sentence2 should start with same letter as sentence1"
# sentence2 should be unique
assert df['sentence2'].is_unique, "sentence2 should be unique"
# label should be all 1
assert all(df['label'] == 1), "label should be all 1"
# number of elements, a:1, b:2, c:3, d:4
assert df.groupby('sentence1').size().to_dict() == {'a': 1, 'b': 2, 'c': 3, 'd': 4}, "Should have correct distribution"
print("- No element limit test passed")

# Test PositiveDataset with max_elements=2 (reduced limit)
print("Testing PositiveDataset with reduced element limit...")
pos_ds = PositiveDataset(
    target_concepts=target_concepts_comp,
    name_table=name_table_comp,
    name_bridge=name_bridge_comp,
    max_elements=2
)

df1 = pd.DataFrame(iter(pos_ds))
# sentence2 should start with the letter in sentence1
assert all(df1['sentence1'].str[0] == df1['sentence2'].str[0]), "sentence2 should start with same letter as sentence1"
# sentence2 should be unique
assert df1['sentence2'].is_unique, "sentence2 should be unique"
# label should be all 1
assert all(df1['label'] == 1), "label should be all 1"
# number of elements, a:1, b:2, c:2, d:2
assert df1.groupby('sentence1').size().to_dict() == {'a': 1, 'b': 2, 'c': 2, 'd': 2}, "Should respect max_elements limit"
print("- Reduced element limit test passed")

# Test PositiveDataset with seed for reproducibility
print("Testing PositiveDataset with seed...")
pos_ds_random = PositiveDataset(
    target_concepts=target_concepts_comp,
    name_table=name_table_comp,
    name_bridge=name_bridge_comp,
    max_elements=2,
    seed=42
)

df2 = pd.DataFrame(iter(pos_ds_random))
# sentence2 should start with the letter in sentence1
assert all(df2['sentence1'].str[0] == df2['sentence2'].str[0]), "sentence2 should start with same letter as sentence1"
# sentence2 should be unique
assert df2['sentence2'].is_unique, "sentence2 should be unique"
# label should be all 1
assert all(df2['label'] == 1), "label should be all 1"
# number of elements, a:1, b:2, c:2, d:2
assert df2.groupby('sentence1').size().to_dict() == {'a': 1, 'b': 2, 'c': 2, 'd': 2}, "Should have same distribution"

# After sorting, df1 should not be equal to df2 (different randomization)
assert not df1.sort_values(['sentence1', 'sentence2']).equals(df2.sort_values(['sentence1', 'sentence2'])), "Different seeds should produce different results"
print("- Seed-based randomization test passed")

print("\nStarting comprehensive NegativeDataset tests...")

# Test NegativeDataset with large max_elements
print("Testing NegativeDataset with no element limit...")
neg_ds = NegativeDataset(
    target_concepts=target_concepts_comp,
    name_table=name_table_comp,
    blacklist_bridge=name_bridge_comp, 
    max_elements=99, 
    seed=42
)

df = pd.DataFrame(iter(neg_ds))
# sentence2 should not start with the letter in sentence1 (since we exclude matching pairs)
assert not all(df['sentence1'].str[0] == df['sentence2'].str[0]), "Should have non-matching pairs"
# number of elements: a: len - 1, b: len - 2, c: len - 3, d: len - 4
expected = {
    'a': len(name_table_comp) - 1,  # 10 - 1 = 9
    'b': len(name_table_comp) - 2,  # 10 - 2 = 8  
    'c': len(name_table_comp) - 3,  # 10 - 3 = 7
    'd': len(name_table_comp) - 4   # 10 - 4 = 6
}
assert df.groupby('sentence1').size().to_dict() == expected, "Should have correct negative distribution"
print("- No element limit negative test passed")

# Test NegativeDataset with reduced max_elements
print("Testing NegativeDataset with reduced element limit...")
neg_ds = NegativeDataset(
    target_concepts=target_concepts_comp,
    name_table=name_table_comp,
    blacklist_bridge=name_bridge_comp, 
    max_elements=2,  # reduce max_elements
    seed=42
)

df = pd.DataFrame(iter(neg_ds))
# sentence2 should not start with the letter in sentence1
assert not all(df['sentence1'].str[0] == df['sentence2'].str[0]), "Should have non-matching pairs"
# number of elements: a: 2, b: 2, c: 2, d: 2
assert df.groupby('sentence1').size().to_dict() == {'a': 2, 'b': 2, 'c': 2, 'd': 2}, "Should respect max_elements limit"
print("- Reduced element limit negative test passed")

# Test NegativeDataset resample functionality
print("Testing NegativeDataset resample...")
neg_ds_random = neg_ds.resample(seed=44)
df_random = pd.DataFrame(iter(neg_ds_random))
# sentence2 should not start with the letter in sentence1
assert not all(df_random['sentence1'].str[0] == df_random['sentence2'].str[0]), "Should have non-matching pairs after resample"
assert df_random.groupby('sentence1').size().to_dict() == {'a': 2, 'b': 2, 'c': 2, 'd': 2}, "Should maintain same distribution"

# After sorting, df should not be equal to df_random (different randomization)
assert not df.sort_values(['sentence1', 'sentence2']).equals(df_random.sort_values(['sentence1', 'sentence2'])), "Different resampling should produce different results"
print("- Resample functionality test passed")

print("\nStarting comprehensive CombinedDataset tests...")

# Test CombinedDataset with DataFrame inputs
print("Testing CombinedDataset with DataFrame inputs...")
df1_test = pd.DataFrame({
    "a": [1, 2, 3]
})
df2_test = pd.DataFrame({
    "a": [4, 5, 6]
})

cd = CombinedDataset(
    df1=df1_test,
    df2=df2_test
)

df_combined = pd.DataFrame(iter(cd))

assert df_combined.shape == (6, 1), "Combined DataFrame should have 6 rows and 1 column"
assert df_combined['a'].tolist() == [1, 2, 3, 4, 5, 6], "Should maintain order of combination"
print("- DataFrame combination test passed")

# Test CombinedDataset shuffle functionality
print("Testing CombinedDataset shuffle...")
cd_random = cd.shuffle(seed=42)
df_random = pd.DataFrame(iter(cd_random))
assert df_random.shape == (6, 1), "Shuffled DataFrame should maintain same shape"
# After sorting by 'a', it should be equal to the original df
assert df_combined.sort_values('a').reset_index(drop=True).equals(df_random.sort_values('a').reset_index(drop=True)), "Should contain same data after shuffle"
print("- Shuffle functionality test passed")


print("\n All tests passed successfully!")