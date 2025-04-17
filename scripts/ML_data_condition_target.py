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
conceptEX = pd.read_feather('data/omop_feather/conceptEX.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')
concept_ancestor = pd.read_feather('data/omop_feather/concept_ancestor.feather')
conceptML = pd.read_feather('data/ML/base_data/conceptML.feather')


conceptML.columns
# ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
#        'concept_code', 'nonstd_name', 'nonstd_concept_id', 'synonym_name',
#        'description', 'all_nonstd_name', 'all_nonstd_concept_id']



logger.log("Define the target standard concepts")
## Reserved concepts are also in the target concepts!
std_target = conceptML[conceptML['domain_id'] == 'Condition'].reset_index(drop=True)
std_target['std_name'] = get_sentence_name(std_target['domain_id'], std_target['concept_name'])

print(f"std concept #: {len(std_target)}")   # 160288
std_target.columns
# ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
#        'concept_code', 'nonstd_name', 'nonstd_concept_id', 'synonym_name',
#        'description', 'all_nonstd_name', 'all_nonstd_concept_id', 'source',
#        'std_name']



logger.log("Exclude the reserved concepts from the non-standard concepts")
# Exclude the reserved concepts from the non-standard concepts.
# - This will NOT exclude the reserved concepts from the standard concepts as standard concepts are the required component in OMOP.
reserved_vocab = "CIEL"
reserved_concepts = conceptEX[(conceptEX.standard_concept != 'S')&(conceptEX.vocabulary_id == reserved_vocab)]
reserved_concept_ids = set(reserved_concepts.concept_id.to_list())
print(f"reserved concepts #: {len(reserved_concept_ids)}") # 50881



# Calculate the total number of non-standard concepts
def total_nonstd(df):
    nonstd_concept_ids_list = df['all_nonstd_concept_id'].to_list()
    # expand the list
    nonstd_concept_ids_list = [concept_id for sublist in nonstd_concept_ids_list for concept_id in sublist]
    return len(nonstd_concept_ids_list)


logger.log("remove reserved concepts from std-to-nonstd mapping")
print(f"nonstd # before removal: {total_nonstd(std_target)}") # 679651

std_target[['nonstd_concept_id', 'nonstd_name']] = std_target.swifter.apply(
    lambda x: remove_reserved(x, reserved_concept_ids, 'nonstd_concept_id', 'nonstd_name'), axis=1, result_type='expand'
)

std_target[['all_nonstd_concept_id', 'all_nonstd_name', 'source']] = std_target.swifter.apply(
    lambda x: remove_reserved(x, reserved_concept_ids, 'all_nonstd_concept_id', 'all_nonstd_name', 'source'), axis=1, result_type='expand'
)

## total number of non-standard concepts after removing reserved concepts
print(f"nonstd # after removal: {total_nonstd(std_target)}") # 648802

root = "data/ML/base_data"
if not os.path.exists(root):
    os.makedirs(root)

std_target.to_feather(os.path.join(root, "std_target.feather"))
reserved_concepts.to_feather(os.path.join(root, "reserved_concepts.feather"))


logger.done()