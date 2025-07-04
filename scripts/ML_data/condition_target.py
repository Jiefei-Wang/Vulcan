# Define the target concept for model training
#  - `std_target` contains target concepts that everything will be mapped to
#  - `reserved_concepts` is provided for the validation of the model

import os
import pandas as pd
from modules.ML_sampling import get_sentence_name, remove_reserved
import swifter
from modules.timed_logger import logger
logger.reset_timer()


logger.log("Loading concept tables")
concept = pd.read_feather('data/omop_feather/concept.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')
concept_ancestor = pd.read_feather('data/omop_feather/concept_ancestor.feather')
concept_names = pd.read_feather("data/base_data/concept_names.feather")



# Extract standard concept names and descriptions
std_concept = concept[concept['standard_concept'] == 'S']
nonstd_concept = concept[concept['standard_concept'] != 'S']


std_conditions = std_concept[std_concept.domain_id == 'Condition']
std_conditions.groupby('vocabulary_id')['concept_id'].count()
"""
HCPCS                    1
ICDO3                56858
Nebraska Lexicon      1274
OMOP Extension         341
OPCS4                    1
SNOMED               98720
SNOMED Veterinary     3093
"""

nonstd_conditions = nonstd_concept[nonstd_concept.domain_id == 'Condition']
nonstd_conditions.groupby('vocabulary_id')['concept_id'].count()
"""
CDISC                   455
CIEL                  38818
CIM10                 13885
CO-CONNECT               16
Cohort                   66
HemOnc                  260
ICD10                 14113
ICD10CM               88510
ICD10CN               30588
ICD10GM               15952
ICD9CM                14929
ICDO3                  5677
KCD7                  19705
MeSH                  12343
Nebraska Lexicon     150062
OMOP Extension            8
OPCS4                     5
OXMIS                  5704
OncoTree                885
PPI                      74
Read                  47836
SMQ                     324
SNOMED                58172
SNOMED Veterinary       144
"""




concept.columns
# ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
#    'concept_class_id', 'standard_concept', 'concept_code',
#    'valid_start_date', 'valid_end_date', 'invalid_reason']


logger.log("Exclude the reserved concepts from the non-standard concepts")
# Exclude the reserved concepts from the non-standard concepts.
# - This will NOT exclude the reserved concepts from the standard concepts as standard concepts are the required component in OMOP.
reserved_vocab = "CIEL"
reserved_concepts = nonstd_concept[nonstd_concept.vocabulary_id == reserved_vocab]
reserved_concept_ids = set(reserved_concepts.concept_id.to_list())
print(f"reserved concepts #: {len(reserved_concept_ids)}") # 50881


## Remove the reserved concepts from the concept_names
matching_map = concept_names[~concept_names['source_id'].isin(reserved_concepts.concept_id.astype(str))].reset_index(drop=True)
# [11424406 rows x 5 columns]


logger.log("Define the target standard concepts")
## Reserved concepts are also in the target concepts if they are standard!
all_targets = std_concept[std_concept.domain_id == 'Condition'].reset_index(
    drop=True)
# make sure the target concepts have at least one name mapping
all_targets = all_targets[all_targets.concept_id.isin(
    matching_map['concept_id'].unique())].reset_index(drop=True)

# also make sure matching_map only contains the target concepts
matching_map = matching_map.merge(
    all_targets[['concept_id', 'concept_name']],
    on='concept_id',
    how='inner')
# [1325247 rows x 5 columns]


print(f"std concept #: {len(all_targets)}")   # 160288
all_targets.columns
# ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
    #    'concept_class_id', 'standard_concept', 'concept_code',
    #    'valid_start_date', 'valid_end_date', 'invalid_reason']




root = "data/ML"
if not os.path.exists(root):
    os.makedirs(root)
all_targets.to_feather(os.path.join(root, "all_targets.feather"))
reserved_concepts.to_feather(os.path.join(root, "reserved_concepts.feather"))
matching_map.to_feather(os.path.join(root, "matching_map.feather"))


logger.done()