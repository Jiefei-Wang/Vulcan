import pandas as pd
import time
from tqdm import tqdm
import random
from tqdm.auto import tqdm
from modules.ML_sampling import MatchingIterableDataset, OffspringIterableDataset, AncestorIterableDataset, get_sentence_name
from modules.ML_data import get_matching

concept = pd.read_feather('data/omop_feather/concept.feather')
concept_ancestor = pd.read_feather('data/omop_feather/concept_ancestor.feather')

conceptEX = pd.read_feather('data/omop_feather/conceptEX.feather')
        
positive_dataset_matching = pd.read_feather('data/ML/matching/positive_dataset_matching.feather')
candidate_df_matching = pd.read_feather('data/ML/matching/candidate_dataset_matching.feather')


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
pandas.isna() will cause problem? 
"""


data_folder = 'data/ML'
iterable_matching = get_matching(data_folder, n_neg=2, seed=42)

it = iter(iterable_matching)
next(it)


print(positive_dataset_matching.iloc[1000])
"""
sentence1      Condition: Neoplasm, benign of hematopoietic s...
sentence2                                 Lymphoid Benign (LBGN)
concept_id1                                               751726
concept_id2                                               777429
label                                                          1
source                                               nonstd_name
"""


concept_dict = concept.set_index("concept_id")["concept_name"].to_dict()

# Create a blacklist of sentences for each concept_id
blacklist = (
    positive_dataset_matching.groupby("concept_id1")["sentence2"]
    .apply(lambda x: set(x.dropna())) 
    .to_dict()
)

# Create a dictionary of sentences to concept_ids, also store the sentence2 with multiple concept_id1s
sentence_to_ids = positive_dataset_matching.groupby('sentence2')["concept_id1"].agg(set)


"""
Function to check if the label is valid for a given item
associated_ids: set of concept_ids associated with the sentence
is_in_blacklist: boolean, True if the sentence is in the blacklist
is_in_multiple_ids: boolean, True if the sentence has multiple concept_ids
label: int, label value
"""
def check_label_validity(item, label):
    ## remove [MATCHING]
    sentence = item["sentence2"][11:]
    concept_id = item["concept_id1"]
    associated_ids = sentence_to_ids.get(sentence, set())
    is_in_blacklist = sentence in blacklist.get(concept_id, set())
    is_in_multiple_ids = len(associated_ids) > 1
    if label == 1:
        return is_in_blacklist  # Label 1 is valid if in blacklist
    elif label == 0:
        return is_in_multiple_ids or not is_in_blacklist  # Label 0 validity check
    return False


# Check the validity of the labels
incorrect_label_1 = 0
incorrect_label_0 = 0
label1 = 0
label0 = 0
for it in tqdm(iterable_matching):    
    sentence1 = it["sentence1"]
    concept_id1 = it["concept_id1"]
    sentence2 = it["sentence2"]
    concept_id2 = it["concept_id2"]
    expected_sentence1 = concept_dict.get(concept_id1, None)
    expected_sentence2 = concept_dict.get(concept_id2, None)
    assert sentence2.startswith("[MATCHING]")
    cleaned_sentence1 = sentence1[11:]
    cleaned_sentence2 = sentence2[11:]
    assert(cleaned_sentence1 == expected_sentence1), f"concept_id1 does not match concept name: {sentence1} != {expected_sentence1}, {concept_id1}"
    # there are cases that concept_id2 is None, so skip the check with these cases 
    if concept_id2 is not None:
        assert(cleaned_sentence2 == expected_sentence2), f"concept_id2 does not match concept name: {sentence2} != {expected_sentence2}, {concept_id1}, {sentence1}, {concept_id2}"
    label = it["label"]
    # check if the label is valid
    is_valid = check_label_validity(it, label)
    if label == 1:
        label1 += 1
        if not is_valid:
            incorrect_label_1 += 1
    elif label == 0:
        label0 += 1 
        if not is_valid:
            incorrect_label_0 += 1
        
    
# Print separate incorrect counts
print(f"Total incorrect label=1 cases: {incorrect_label_1}")
print(f"Total incorrect label=0 cases: {incorrect_label_0}")
print(f"Total incorrect labels: {incorrect_label_0 + incorrect_label_1}")     
# Total incorrect labels: 0


label0 #2702248 = 675562 * 4
label1 #675562


################
# check the validity of the relation tables 
################

### ancestor 

positive_dataset_relation = pd.read_feather('data/ML/relation/positive_dataset_relation.feather')
std_concept = pd.read_feather('data/omop_feather/concept.feather')

std_concept[std_concept['concept_id']==500000301]
std_concept[std_concept['concept_id']==42596042]
positive_dataset_relation[positive_dataset_relation['concept_id1']==500000301]
print(std_concept.info()) 

candidate_df_relation = std_concept.copy()
candidate_df_relation['std_name'] = get_sentence_name(candidate_df_relation['domain_id'], candidate_df_relation['concept_name'])

myiter_ancestor = AncestorIterableDataset(
    positive_dataset_relation,
    candidate_df_relation
)
positive_dataset_relation.iloc[0]
print(candidate_df_relation.iloc[0])


it = iter(myiter_ancestor)
next(it)
next(it)

sentence_to_ids_ancestor = positive_dataset_relation.groupby('sentence2')["concept_id1"].agg(set)
sentence_to_ids_ancestor.head()

multiple_ids = sentence_to_ids_ancestor[sentence_to_ids_ancestor.apply(lambda ids: len(ids) > 1)]


blacklist_ancestor = (
    positive_dataset_relation.groupby("concept_id1")["sentence2"]
    .apply(lambda x: set(x.dropna())) 
    .to_dict()
)

values = blacklist_ancestor.get(42596042)
if values is not None:
    print("Values for key 40481826:", values)
else:
    print("No values found for key 40481826")


def check_ancestor_label_validity(item, label):
    sentence = item["sentence2"].split(":", 1)[1].strip()
    concept_id = item["concept_id1"]
    associated_ids = sentence_to_ids_ancestor.get(sentence, set())
    
    blacklist_set = blacklist_ancestor.get(concept_id, set())
    normalized_blacklist = {remove_prefix(s) for s in blacklist_set}
    
    is_in_blacklist = sentence in normalized_blacklist
    is_in_multiple_ids = len(associated_ids) > 1
    if label == 1:
        return is_in_blacklist  # Label 1 is valid if in blacklist
    elif label == 0:
        return is_in_multiple_ids or not is_in_blacklist  # Label 0 validity check
    return False


def remove_prefix(text, prefix="Condition:"):
    if text.startswith(prefix):
        return text[len(prefix):].strip()
    return text.strip()


# Check the validity of the labels
incorrect_label_1 = 0
incorrect_label_0 = 0
label1 = 0
label0 = 0
for it in tqdm(myiter_ancestor):    
    sentence1 = it["sentence1"]
    concept_id1 = it["concept_id1"]
    sentence2 = it["sentence2"]
    concept_id2 = it["concept_id2"]
    expected_sentence1 = concept_dict.get(concept_id1, None)
    expected_sentence2 = concept_dict.get(concept_id2, None)
    assert (sentence2.startswith('[ANCESTOR]')), f"Expected sentence2 to start with '[ANCESTOR]': {sentence2}"
    cleaned_sentence1 = sentence1.split(":", 1)[1].lstrip() if ":" in sentence1 else sentence1
    cleaned_sentence2 = sentence2.split(":", 1)[1].lstrip() if ":" in sentence2 else sentence2
    assert(cleaned_sentence1 == expected_sentence1), f"concept_id1 does not match concept name: {cleaned_sentence1} != {expected_sentence1}, {concept_id1}"
    # there are cases that concept_id2 is None, so skip the check with these cases 
    if concept_id2 is not None:
        assert(cleaned_sentence2 == expected_sentence2), f"concept_id2 does not match concept name: {repr(cleaned_sentence2)} != {repr(expected_sentence2)}, {concept_id2}"
    label = it["label"]
    # blacklist_set = blacklist_ancestor.get(concept_id1, set())
    # normalized_blacklist = {remove_prefix(s) for s in blacklist_set}
    is_valid = check_ancestor_label_validity(it, label)
    if label == 1:
        label1 += 1
        if not is_valid:
            incorrect_label_1 += 1
    elif label == 0:
        label0 += 1 
        if not is_valid:
            incorrect_label_0 += 1
        
    
    
value = blacklist_ancestor.get(42596042, "Key not found")
print("Value for key 37110208:", value)


# Print separate incorrect counts
print(f"Total incorrect label=1 cases: {incorrect_label_1}")
print(f"Total incorrect label=0 cases: {incorrect_label_0}")
print(f"Total incorrect labels: {incorrect_label_0 + incorrect_label_1}")     
# Total incorrect labels: 0

### offspring 

reserved_vocab = "CIEL"
conceptML = pd.read_feather('data/ML/conceptML1.feather')
conceptML.columns
"""
['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
       'concept_code', 'nonstd_name', 'nonstd_concept_id', 'synonym_name',
       'descriptions'],
      dtype='object')
"""
conceptML['domain_id'].unique()
"""
['Provider' 'Device' 'Observation' 'Metadata' 'Visit' 'Drug' 'Measurement'
 'Condition Status' 'Cost' 'Currency' 'Episode' 'Ethnicity' 'Gender'
 'Procedure' 'Condition' 'Regimen' 'Spec Anatomic Site' 'Payer'
 'Revenue Code' 'Note' 'Meas Value' 'Language' 'Specimen' 'Geography'
 'Plan' 'Plan Stop Reason' 'Race' 'Relationship' 'Route'
 'Meas Value Operator' 'Sponsor' 'Type Concept' 'Unit']
"""
std_concept = conceptML.copy()
std_concept['std_name'] = get_sentence_name(conceptML['domain_id'], conceptML['concept_name'])


std_target = conceptML[(conceptML['domain_id'] == 'Condition')& (conceptML['vocabulary_id'] != reserved_vocab)].reset_index(drop=True)
std_target['std_name'] = get_sentence_name(std_target['domain_id'], std_target['concept_name'])
print(len(std_target))   # 160288
print(std_target.columns)
"""
['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
       'concept_code', 'nonstd_name', 'nonstd_concept_id', 'synonym_name',
       'descriptions', 'std_name'],
      dtype='object')
"""
