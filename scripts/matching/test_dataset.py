import pandas as pd
from modules.Dataset import *

with open('modules/Dataset.py') as f:
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


# Test positive 
pos_ds = PositiveDataset(
    target_concepts=target_concepts,
    name_table=name_table,
    name_bridge=name_bridge,
    max_elements=10
)

df = pd.DataFrame(iter(pos_ds))
# sentence2 should start with the letter in sentence1
assert all(df['sentence1'].str[0] == df['sentence2'].str[0])
# sentence2 should be unique
assert df['sentence2'].is_unique
# label should be all 1
assert all(df['label'] == 1)
# number of elements, a:1, b:2, c:3, d:4
assert df.groupby('sentence1').size().to_dict() == {'a': 1, 'b': 2, 'c': 3, 'd': 4}

## reduce max_element 

pos_ds = PositiveDataset(
    target_concepts=target_concepts,
    name_table=name_table,
    name_bridge=name_bridge,
    max_elements=2
)

df = pd.DataFrame(iter(pos_ds))
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
neg_ds = NegativeDataset(
    target_concepts=target_concepts,
    name_table=name_table,
    blacklist_bridge=name_bridge, 
    max_elements=99, 
    seed=42
)

df = pd.DataFrame(iter(neg_ds))
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


neg_ds = NegativeDataset(
    target_concepts=target_concepts,
    name_table=name_table,
    blacklist_bridge=name_bridge, 
    max_elements=2,  # reduce max_elements
    seed=42
)

df = pd.DataFrame(iter(neg_ds))
# sentence2 should not start with the letter in sentence1
assert not all(df['sentence1'].str[0] == df['sentence2'].str[0])
# number of elements: a: 2, b: 2, c: 2, d: 2
assert df.groupby('sentence1').size().to_dict() == {'a': 2, 'b': 2, 'c': 2, 'd': 2}


df1 = pd.DataFrame({
    "a": [1, 2, 3]
})
df2 = pd.DataFrame({
    "a": [4, 5, 6]
})

cd = CombinedDataset(
    df1=df1,
    df2=df2
)

df = pd.DataFrame(iter(cd))

assert df.shape == (6, 1)  
assert df['a'].tolist() == [1, 2, 3, 4, 5, 6] 


cd[:5]