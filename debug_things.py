import pandas as pd
 # Load the Excel file
df = pd.read_excel("verify_dataset/Condition Matching Test.xlsx")

# Check data types
print("Data types:")
print(df.dtypes)

# Check for non-numeric values in ID columns
def check_non_numeric_ids(df, id_cols):
    for col in id_cols:
        if col in df.columns:
            # Try to convert to numeric and see what fails
            numeric_version = pd.to_numeric(df[col], errors='coerce')
            non_numeric_count = numeric_version.isna().sum()
            print(f"\n{col}: {non_numeric_count} non-numeric values")

            if non_numeric_count > 0:
                # Show examples of non-numeric values
                non_numeric_rows = df[numeric_version.isna()]
                print(f"Examples of non-numeric {col}:")
                print(non_numeric_rows[[col]].head())

# Check ID columns
id_columns = ['query_id', 'corpus_id']
check_non_numeric_ids(df, id_columns)



concept_table = pd.read_feather('data/omop_feather/concept.feather')
concept_table.iloc[0]

matching_map_table = pd.read_feather('data/matching/matching_map_table.feather')
matching_map_table.iloc[1000]
matching_map_table.columns

excel_file =  "verify_dataset/Condition Matching Test.xlsx"
df = pd.read_excel(excel_file)
df.columns
df.shape[0]

matching_map_table['source_id_int'] = pd.to_numeric(matching_map_table['source_id'], errors='coerce')

print("Checking for NaN values after conversion:")
nan_count = matching_map_table['source_id_int'].isna().sum()
print(f"NaN values in source_id_int: {nan_count}")

matching_map_clean = matching_map_table[matching_map_table['source_id_int'].notna()].copy()
matching_map_clean['source_id_int'] = matching_map_clean['source_id_int'].astype('int64')
print(f"Original matching_map_table: {len(matching_map_table)} rows")
print(f"Clean matching_map_table: {len(matching_map_clean)} rows")

matching_pairs = set(zip(matching_map_clean['source_id_int'], matching_map_clean['concept_id']))
print(f"Total matching pairs: {len(matching_pairs)}")


 # Verify labels in Excel file
# Verify labels in Excel file
def verify_labels(df, matching_pairs):
    correct_labels = 0
    incorrect_labels = 0
    for idx, row in df.iterrows():
        query_id = row['query_id']
        corpus_id = row['corpus_id']
        label = row['label']
        pair_exists = (query_id, corpus_id) in matching_pairs
        expected_label = 1 if pair_exists else 0
        if label == expected_label:
            correct_labels += 1
        else:
            incorrect_labels += 1
            if incorrect_labels <= 10:  # Show first 10 errors
                print(f"Row {idx}: query_id={query_id}, corpus_id={corpus_id}, label={label}, expected={expected_label}")
    print(f"\nCorrect labels: {correct_labels}")
    print(f"Incorrect labels: {incorrect_labels}")
    print(f"Accuracy: {correct_labels/(correct_labels+incorrect_labels)*100:.2f}%")

verify_labels(df, matching_pairs)

45905817
matching_map_clean[matching_map_clean['source_id_int'] == 36527342]
matching_map_clean[matching_map_clean['concept_id'] == 36527342]

matching_map_clean[matching_map_clean['source_id_int'] == 1128]


  # Get total counts
total_rows = len(matching_map_table)
mismatch_count = (matching_map_table['concept_id'] != matching_map_table['source_id']).sum()
match_count = total_rows - mismatch_count

print(f"Total rows: {total_rows}")
print(f"Matching concept_id == source_id: {match_count}")
print(f"Mismatched concept_id != source_id: {mismatch_count}")
print(f"Percentage mismatched: {mismatch_count/total_rows*100:.2f}%")

# Show actual mismatched examples
true_mismatches = matching_map_table[matching_map_table['concept_id'] != matching_map_table['source_id']]
print("\nActual mismatched examples:")
print(true_mismatches[['concept_id', 'source_id', 'name']].head(10))

# Show some matching examples for comparison
matches = matching_map_table[matching_map_table['concept_id'] == matching_map_table['source_id']]
print("\nMatching examples:")
print(matches[['concept_id', 'source_id', 'name']].head(5))


concept_ancestor_table = pd.read_feather('data/omop_feather/concept_ancestor.feather')

# 76154
concept_ancestor_table.dtypes



concept_ancestor_table[concept_ancestor_table['ancestor_concept_id'] == 437312]

concept_ancestor_table[
    (concept_ancestor_table['ancestor_concept_id'] == 437312) &
    (concept_ancestor_table['descendant_concept_id'] == 76154)
]

concept_ancestor_table[
    (concept_ancestor_table['ancestor_concept_id'] == 76154) &
    (concept_ancestor_table['descendant_concept_id'] == 437312)
]


# condition_matching_name_table_train.feather
condition_matching_name_table_train_table = pd.read_feather('condition_matching_name_table_train.feather')
condition_matching_name_table_train_table.iloc[0]


name_table = pd.read_feather('data/matching/condition_matching_name_table_train.feather')
print(condition_matching_name_table_train_table['type'].unique())

condition_matching_name_table_train_table[condition_matching_name_table_train_table['name_id'] == 2824]
matching_map_clean[matching_map_clean['source_id_int'] == 45586598]


row = condition_matching_name_table_train_table[condition_matching_name_table_train_table['name_id'] == 4474]
for index, data in row.iterrows():
    print(f"name_id: {data['name_id']}")
    print(f"source: {data['source']}")
    print(f"source_id: {data['source_id']}")
    print(f"type: {data['type']}")
    print(f"name: {data['name']}")
    
    
    
matching_map_clean[matching_map_clean['source_id_int'] == 40390208]

condition_matching_name_table_train_table[condition_matching_name_table_train_table['name_id'] == 83062]



matching_map_clean[matching_map_clean['source_id_int'] == 40401899]

matching_map_clean[matching_map_clean['source_id_int'] == 4110106]

import pandas as pd
df = pd.read_feather('data/matching/condition_matching_test_pos.feather')
print('Columns:', list(df.columns))
print('Shape:', df.shape)
print('Sample data:')
print(df.head())
print('Labels distribution:')
print(df['label'].value_counts())