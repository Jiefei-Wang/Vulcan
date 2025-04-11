import pandas as pd



print("Sanity check for conceptML")
conceptML = pd.read_feather('data/ML/ML_data/conceptML.feather')

print(f"Dataset dim: {conceptML.shape}")

print("Checking for NA values in conceptML")
total_na = conceptML['concept_id'].isna().sum() +\
conceptML['concept_name'].isna().sum() +\
conceptML['domain_id'].isna().sum() +\
conceptML['vocabulary_id'].isna().sum() +\
conceptML['concept_code'].isna().sum()

if total_na > 0:
    raise ValueError(f"Total NA values in conceptML: {total_na}")


def check_list_length(row, col1, col2):
    if len(row[col1]) != len(row[col2]):
        raise ValueError(f"Length of {col1} and {col2} do not match at index {row.name}")


print("Check if the length matches")
## len(nonstd_name) == len(nonstd_concept_id)
conceptML.apply(
    lambda x: check_list_length(x, 'nonstd_name', 'nonstd_concept_id'), axis=1
)
## len(all_nonstd_concept_id)== len(all_nonstd_concept_id)
conceptML.apply(
    lambda x: check_list_length(x, 'all_nonstd_name', 'all_nonstd_concept_id'), axis=1
)
    
# for each row, the first len(nonstd_name) elements of all_nonstd_concept_id should be non-na, the rest should be na
# for i in range(len(conceptML)):
#     nonstd_name = conceptML['nonstd_name'][i]
#     all_nonstd_concept_id = conceptML['all_nonstd_concept_id'][i]
#     if len(nonstd_name) > 0 and any(pd.isna(all_nonstd_concept_id[:len(nonstd_name)])):
#         raise ValueError(f"Unexpected values in all_nonstd_concept_id at index {i}")
#     if all(pd.isna(all_nonstd_concept_id[len(nonstd_name):])) == False:
#         raise ValueError(f"Unexpected values in all_nonstd_concept_id at index {i}")

print("Total elements in each column")
for col in ['nonstd_name', 'nonstd_concept_id','synonym_name', 'description', 'all_nonstd_name', 'all_nonstd_concept_id']:
    if col in conceptML.columns:
        list_lengths = conceptML[col].apply(len)
        print(f"{col}: {list_lengths.sum()}")



mem = conceptML.memory_usage(deep=True)
for col, size in mem.items():
    print(f"{col}: {size / 1024**2:.2f} MB")
# Index: 0.00 MB
# concept_id: 29.96 MB
# domain_id: 3.33 MB
# vocabulary_id: 3.33 MB
# concept_code: 223.91 MB
# nonstd_name: 246.78 MB
# nonstd_concept_id: 246.78 MB
# synonym_name: 240.00 MB
# description: 215.25 MB
# all_nonstd_name: 281.90 MB
# all_nonstd_concept_id: 255.33 MB

for col, dtype in conceptML.dtypes.items():
    print(f"{col}: {dtype}")
# concept_id: Int64
# domain_id: category
# vocabulary_id: category
# concept_code: string
# nonstd_name: object
# nonstd_concept_id: object
# synonym_name: object
# description: object
# all_nonstd_name: object
# all_nonstd_concept_id: object

