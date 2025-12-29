import pandas as pd
import os
from modules.timed_logger import logger
from sklearn.model_selection import train_test_split
from modules.CodeBlockExecutor import trace, tracedf

logger.reset_timer()
logger.log("loading data")

output_dir = "data/matching"
output_train_dir = os.path.join(output_dir, 'train')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_train_dir):
    os.makedirs(output_train_dir)


# std_bridge = pd.read_feather("data/omop_feather/std_bridge.feather")
concept= pd.read_feather('data/omop_feather/concept.feather')
# matching_map_table = pd.read_feather('data/matching/matching_map_table.feather')

condition_matching_pairs = pd.read_feather('data/matching/condition_matching_pairs.feather')



####################
## Train, test, valid split
####################
logger.log("Train, test, valid split")

test_num = 5000
valid_num = 5000

indices = condition_matching_pairs.index.tolist()
train_idx, temp_idx = train_test_split(
    indices, test_size=test_num + valid_num, random_state=42, shuffle=True
)
valid_idx, test_idx = train_test_split(
    temp_idx, test_size=test_num, random_state=42, shuffle=True
)

matching_pairs_train = condition_matching_pairs.loc[train_idx].reset_index(drop=True)
matching_pairs_valid = condition_matching_pairs.loc[valid_idx].reset_index(drop=True)
matching_pairs_test = condition_matching_pairs.loc[test_idx].reset_index(drop=True)

matching_pairs_train.to_feather(os.path.join(output_train_dir, 'matching_pairs_train.feather'))
matching_pairs_valid.to_feather(os.path.join(output_train_dir, 'matching_pairs_valid.feather'))
matching_pairs_test.to_feather(os.path.join(output_train_dir, 'matching_pairs_test.feather'))

tracedf(matching_pairs_train)
#> DataFrame dimensions: 535158 rows × 7 columns
#> Column names:
#> ['concept_id', 'concept_name', 'source', 'source_id', 'type', 'name_id', 'name']
#> Estimated memory usage: 205.11 MB

tracedf(matching_pairs_valid)
#> DataFrame dimensions: 5000 rows × 7 columns
#> Column names:
#> ['concept_id', 'concept_name', 'source', 'source_id', 'type', 'name_id', 'name']
#> Estimated memory usage: 1.92 MB

tracedf(matching_pairs_test)
#> DataFrame dimensions: 5000 rows × 7 columns
#> Column names:
#> ['concept_id', 'concept_name', 'source', 'source_id', 'type', 'name_id', 'name']
#> Estimated memory usage: 1.91 MB
