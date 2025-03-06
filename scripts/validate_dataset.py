import pandas as pd
import time
from tqdm import tqdm
import random
from tqdm.auto import tqdm
from modules.ML_sampling import MatchingIterableDataset


concept = pd.read_feather('data/omop_feather/concept.feather')
concept_ancestor = pd.read_feather('data/omop_feather/concept_ancestor.feather')
        
positive_dataset_matching = pd.read_feather('data/ML/matching/positive_dataset_matching.feather')
candidate_df_matching = pd.read_feather('data/ML/matching/candidate_dataset_matching.feather')


concept_dict = concept.set_index("concept_id")["concept_name"].to_dict()

print(positive_dataset_matching.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 675562 entries, 0 to 675561
Data columns (total 6 columns):
 #   Column       Non-Null Count   Dtype  
---  ------       --------------   -----  
 0   sentence1    675562 non-null  object 
 1   sentence2    675562 non-null  object 
 2   concept_id1  675562 non-null  int64  
 3   concept_id2  440913 non-null  float64
 4   label        675562 non-null  int64  
 5   source       675562 non-null  object 
dtypes: float64(1), int64(2), object(3)

cocept_id2 has missing values and float type 
"""

positive_dataset_matching["concept_id2"] = positive_dataset_matching["concept_id2"].astype("Int64")

"""
The NAN becomes <NA> which is a pandas specific type for missing values
will this cause problem?
if cause problem, can use .fillna(-1) to fill the missing values
"""


iterable_matching = MatchingIterableDataset(
    positive_dataset_matching,
    candidate_df_matching,
    n_neg=4,  
    seed=42
)



# no mismatch between id and name
output_file = "mismatches.txt"
with open(output_file, "w") as f:
    f.write("Mismatched Sentences & Concept IDs\n")
    f.write("=" * 50 + "\n")
    for sample in tqdm(iterable_matching, desc="Checking dataset", unit="sample"):
        sentence1 = sample["sentence1"]
        sentence1 = sentence1[11:]
        sentence2 = sample["sentence2"]
        concept_id1 = sample["concept_id1"]
        concept_id2 = sample["concept_id2"]
        clean_sentence2 = sentence2[11:]
        expected_sentence1 = concept_dict.get(concept_id1, None)
        expected_sentence2 = concept_dict.get(concept_id2, None)
        mismatch_found = False  
        if expected_sentence1 and sentence1 != expected_sentence1:
            mismatch_found = True
            mismatch_info = (
                f"Mismatch! sentence1 does not match concept_id1:\n"
                f"  - Found: '{sentence1}'\n"
                f"  - Expected: '{expected_sentence1}'\n"
                f"  - Concept ID: {concept_id1}\n"
                f"{'-' * 50}\n"
            )
            f.write(mismatch_info) 
        if expected_sentence2 and clean_sentence2 != expected_sentence2:
            mismatch_found = True
            mismatch_info = (
                f"Mismatch! sentence2 does not match concept_id2:\n"
                f"  - Found: '{clean_sentence2}'\n"
                f"  - Expected: '{expected_sentence2}'\n"
                f"  - Concept ID: {concept_id2}\n"
                f"{'-' * 50}\n"
            )
            f.write(mismatch_info)
    if mismatch_found:
        print(f"Mismatches found and saved to {output_file}!")
    else:
        print("\nNo mismatches found!")



blacklist = (
    positive_dataset_matching.groupby("concept_id1")["concept_id2"]
    .apply(lambda x: {int(v) if pd.notna(v) else None for v in x})  # Convert NaNs to None
    .to_dict()
)
print(positive_dataset_matching.iloc[1000])
"""
sentence1      Condition: Neoplasm, benign of hematopoietic s...
sentence2                                 Lymphoid Benign (LBGN)
concept_id1                                               751726
concept_id2                                               777429
label                                                          1
source                                               nonstd_name
"""

## Check sample size
# label0= 0
# label1= 0
# for it in tqdm(iterable_matching):
#     ## TODO: check if concept_id matchs concept name
#     # no mismatch
#     sentence1 = it['sentence1']
#     concept_id1 = it['concept_id1']
#     sentence2 = it['sentence2']
#     concept_id2 = it['concept_id2']
#     label = it['label']
#     bl = blacklist.get(concept_id1)
#     ## check if sentence2 has a [MATCHING] label in the beginning
#     assert sentence2.startswith("[MATCHING]")
#     sentence2 = sentence2[11:]
    
#     if label == 1:
#         if concept_id2 is None and not bl:
#             pass
#         elif concept_id2 not in bl:
#             print(f"sentence1: {sentence1}")
#             print(f"concept_id1: {concept_id1}")    
#             print(f"sentence2: {sentence2}")
#             print(f"concept_id2: {concept_id2}")
#             print("*"*50)
#             print(bl)
#             print(f"\n {label}")
#             raise AssertionError(f"sentence2 '{sentence2}' is not in blacklist for sentence1 '{sentence1}'")
#         label1 += 1
#     else:
#         # false 
#         if concept_id2 is None and not bl:
#             pass
#         elif concept_id2 in bl:
#             print(f"sentence1: {sentence1}")
#             print(f"concept_id1: {concept_id1}")
#             print(f"sentence2: {sentence2}")
#             print(f"concept_id2: {concept_id2}")
#             print("*"*50)
#             print(bl)
#             print(f"\n {label}")
#             raise AssertionError(f"sentence2 '{sentence2}' is in blacklist for sentence1 '{sentence1}'")        
#         label0 += 1

label0 = 0
label1 = 0

for it in tqdm(iterable_matching, desc="Checking dataset", unit="sample"):
    sentence1 = it["sentence1"]
    concept_id1 = it["concept_id1"]
    sentence2 = it["sentence2"]
    concept_id2 = it["concept_id2"]
    label = it["label"]
    bl = blacklist.get(concept_id1)
    assert sentence2.startswith("[MATCHING]"), f"sentence2 does not start with [MATCHING]: {sentence2}"
    sentence2 = sentence2[11:]  # Remove "[MATCHING] " prefix
    if label == 1:
        assert concept_id2 in bl
        label1 += 1
    else:
        assert concept_id2 not in bl
        label0 += 1


label0
label1
