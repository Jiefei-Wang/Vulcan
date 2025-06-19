from modules.Iterable import PositiveIterable, NegativeIterable, FalsePositiveIterable, CombinedIterable
import pandas as pd

with open('modules/Iterable.py') as f:
    exec(f.read())

target_concepts = pd.DataFrame({
    'concept_id': [101, 102, 103, 104],
    'concept_name': ['a', 'b', 'c', 'd']
})

name_table = pd.DataFrame({
    'name_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'name': ['a1', 'b1', 'b2', 'c1', 'c2', 'c3', 'd1', 'd2', 'd3', 'd4']
})

name_bridge = pd.DataFrame({
    'concept_id': [101, 102,102,103,103,103,104,104,104,104],
    'name_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
})


# Test positive iterable
print("Testing PositiveIterable:")
pos_it = PositiveIterable(
    target_concepts=target_concepts,
    name_table=name_table,
    name_bridge=name_bridge,
    max_element=5
)

df = pd.DataFrame(pos_it)
# sentence2 should start with the letter in sentence1
assert all(df['sentence1'].str[0] == df['sentence2'].str[0])
# sentence2 should be unique
assert df['sentence2'].is_unique
# label should be all 1
assert all(df['label'] == 1)
# number of elements, a:1, b:2, c:3, d:4
assert df.groupby('sentence1').size().to_dict() == {'a': 1, 'b': 2, 'c': 3, 'd': 4}

## reduce max_element 
pos_it2 = PositiveIterable(
    target_concepts=target_concepts,
    name_table=name_table,
    name_bridge=name_bridge,
    max_element=2
)

df = pd.DataFrame(pos_it2)
# sentence2 should start with the letter in sentence1
assert all(df['sentence1'].str[0] == df['sentence2'].str[0])
# sentence2 should be unique
assert df['sentence2'].is_unique
# label should be all 1
assert all(df['label'] == 1)
# number of elements, a:1, b:2, c:2, d:2
assert df.groupby('sentence1').size().to_dict() == {'a': 1, 'b': 2, 'c': 2, 'd': 2}

############################
## Test NegativeIterable
############################
neg_it = NegativeIterable(
    target_concepts=target_concepts,
    name_table=name_table,
    blacklist_name_bridge=name_bridge, 
    max_element=99, 
    seed=42
)

df = pd.DataFrame(neg_it)
# sentence2 should not start with the letter in sentence1
assert not all(df['sentence1'].str[0] == df['sentence2'].str[0])
# number of elements: a: len - 1, b: len - 2, c: len - 3, d: len - 4
expected = {
    'a': len(name_table) - 1,
    'b': len(name_table) - 2,
    'c': len(name_table) - 3,
    'd': len(name_table) - 4
}
assert df.groupby('sentence1').size().to_dict() == expected


neg_it2 = NegativeIterable(
    target_concepts=target_concepts,
    name_table=name_table,
    blacklist_name_bridge=name_bridge, 
    max_element=2, # Reduced max 
    seed=42
)

df = pd.DataFrame(neg_it2)
# sentence2 should not start with the letter in sentence1
assert not all(df['sentence1'].str[0] == df['sentence2'].str[0])
# number of elements: a: 2, b: 2, c: 2, d: 2
assert df.groupby('sentence1').size().to_dict() == {'a': 2, 'b': 2, 'c': 2, 'd': 2}


############################
## Test FalsePositiveIterable
############################
false_positive_name_bridge = pd.DataFrame({
    'concept_id': [101, 102, 102, 103, 103, 103],
    'name_id': [2, 4, 5, 1,2,3]  # Example false positive names
})

fp_it = FalsePositiveIterable(
    target_concepts=target_concepts,
    name_table=name_table,
    name_bridge=false_positive_name_bridge,
    max_element=99
)

df = pd.DataFrame(fp_it)

# len: a: 1, b: 2, c: 3
assert df.groupby('sentence1').size().to_dict() == {'a': 1, 'b': 2, 'c': 3}
# pairs a-b1, b-c1,b-c2, c-a1,c-b1,c-b2
expected = pd.DataFrame({
    'sentence1': ['a', 'b', 'b', 'c', 'c', 'c'],
    'sentence2': ['b1', 'c1', 'c2', 'a1', 'b1', 'b2'],
    'concept_id1': [101, 102, 102, 103, 103, 103],
    'concept_id2': [2, 4, 5, 1, 2, 3],
    'label': [0, 0, 0, 0, 0, 0]
})
pd.testing.assert_frame_equal(df, expected, check_like=True)


############################
## Test CombinedIterable
############################
com_it = CombinedIterable(
    target_concepts=target_concepts,
    name_table=name_table,
    positive_name_bridge=name_bridge,
    blacklist_name_bridge = name_bridge,
    false_positive_name_bridge=false_positive_name_bridge,
    positive_max_element = 2,
    false_positive_max_element = 2,
    negative_max_element = 2
)

df = pd.DataFrame(com_it)
# len: a: 1+2+1, b: 2+2+2, c: 2+2+2, d: 2+2+0
assert df.groupby('sentence1').size().to_dict() == {'a': 4, 'b': 6, 'c': 6, 'd': 4}
