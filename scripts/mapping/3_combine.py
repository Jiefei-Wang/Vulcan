# Combine all mapping into one file
# File: data/base_data/concept_names.feather

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
# import swifter
from modules.timed_logger import logger
import duckdb

logger.reset_timer()
logger.log("Combining all name maps and tables")


base_path = "data/mapping_data"
map_tables = [
    'map_table_umls.feather',
    'map_table_OMOP.feather',
]


# Load all name maps and tables
matching_map_table = pd.concat(
    [pd.read_feather(os.path.join(base_path, name_map_table)) for name_map_table in map_tables],
    ignore_index=True
)
# [10172534 rows x 5 columns]



matching_map_table.groupby(['source', 'type'])['name'].count()
# OMOP    nonstd     2335383
#         synonym    2851614
# UMLS    DEF         638595
#         STR        4346917
# Name: name, dtype: int64

matching_map_table['name'] = matching_map_table['name'].str.lower()
matching_map_table = matching_map_table.drop_duplicates(['concept_id', 'name']).reset_index(drop=True)
# [8967080 rows x 6 columns]

matching_map_table['source_id'] = matching_map_table['source_id'].astype(str)


####################
## seperate name from mapping
####################
logger.log("Creating name_id and matching_name_bridge tables")

matching_map_table['name_id'] = np.arange(len(matching_map_table)) + 1

matching_name_bridge = matching_map_table[['concept_id', 'name_id']]
matching_name_table = matching_map_table[['name_id', 'source', 'source_id', 'type', 'name']].reset_index(drop=True)


matching_name_bridge.to_feather(os.path.join(base_path, 'matching_name_bridge.feather'))
matching_name_table.to_feather(os.path.join(base_path, 'matching_name_table.feather'))
logger.done()

